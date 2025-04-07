#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.interact.configuration_interact import InterACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize

class InterACTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "act"],
):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    name = "interact"

    def __init__(
        self,
        config: InterACTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = InterACTConfig()
        self.config: InterACTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = InterACT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        l2_loss = (
            F.mse_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item(), "l2_loss": l2_loss.item()}
        # loss_dict["loss"] = 0.8 * l1_loss + 0.2 * l2_loss
        loss_dict["loss"] = l1_loss

        return loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

class InterACT(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self.use_av_arm = config.use_av_arm
        self.num_cls_tokens_arm = config.num_cls_tokens_arm
        self.num_cls_tokens_image = config.num_cls_tokens_image

        if self.use_robot_state:
            num_cls_tokens_arm = config.num_cls_tokens_arm
            self.cls_input_arm1 = nn.Embedding(1, config.dim_model).to(device='cuda:0')
            self.cls_input_arm2 = nn.Embedding(1, config.dim_model).to(device='cuda:0')
            if self.use_av_arm:
                self.cls_input_av = nn.Embedding(1, config.dim_model).to(device='cuda:0')
        
        if self.use_images:
            num_cls_tokens_image = config.num_cls_tokens_image
            self.cls_input_image = nn.Embedding(1, config.dim_model).to(device='cuda:0')

        if self.use_robot_state:
            num_arm_input_token_encoder = num_cls_tokens_arm + 7

            cls_input_arm1 = self.cls_input_arm1.weight
            self.cls_token_arm1 = cls_input_arm1.repeat(num_cls_tokens_arm, 1)
            cls_input_arm2 = self.cls_input_arm2.weight
            self.cls_token_arm2 = cls_input_arm2.repeat(num_cls_tokens_arm, 1)
           
            self.register_buffer(
                "arm1_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
            )
            self.register_buffer(
                "arm2_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
            )
            if self.use_av_arm:
                cls_input_av = self.cls_input_av.weight
                self.cls_token_av = cls_input_av.repeat(num_cls_tokens_arm, 1)  # (num_cls_right, 1, dim_model)
                self.register_buffer(
                    "av_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
                )
        
        if self.use_images:
            cls_input_image = self.cls_input_image.weight
            self.cls_token_image = cls_input_image.repeat(num_cls_tokens_image, 1)  # (num_cls_image, 1, dim_model)
            self.register_buffer(
                    "image_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_cls_tokens_image, config.dim_model).unsqueeze(0),
                )
            
        if self.use_av_arm:
            self.register_buffer(
                "cls_encoder_pos_enc",
                create_sinusoidal_pos_embedding(3*num_cls_tokens_arm + num_cls_tokens_image, config.dim_model).unsqueeze(0),
            )
        else:
            self.register_buffer(
                "cls_encoder_pos_enc",
                create_sinusoidal_pos_embedding(2*num_cls_tokens_arm + num_cls_tokens_image, config.dim_model).unsqueeze(0),
            )
        
        # Backbone for image feature extraction.
        if self.use_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = InterACTEncoder(config)
        self.decoder = ACTDecoder(config)

        self.encoder_robot_state_input_proj = nn.Linear(1, 512)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].


        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        # n_1d_tokens = 1  # for the latent
        # if self.use_robot_state:
        #     n_1d_tokens += 1
        # if self.use_env_state:
        #     n_1d_tokens += 1
        # self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        # Prepare the latent for input to the transformer encoder.
        # if self.config.use_vae and "action" in batch:
        # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
        # cls_embed = einops.repeat(
        #     self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        # )  # (B, 1, D)

        # Prepare transformer encoder inputs.
        arm1_state = batch["observation.state"][:, :7]  # Assuming the first 7 are for the left arm
        arm1_state_proj = self.encoder_robot_state_input_proj(arm1_state.unsqueeze(-1)) 
        arm2_state = batch["observation.state"][:, 7:14]  # Assuming the last 7 are for the right arm
        arm2_state_proj = self.encoder_robot_state_input_proj(arm2_state.unsqueeze(-1))
        if self.use_av_arm:
            av_state = batch["observation.state"][:, 14:21]
            av_state_proj = self.encoder_robot_state_input_proj(av_state.unsqueeze(-1))

        encoder_in_pos_embed = list(self.arm1_encoder_pos_enc)
        encoder_in_pos_embed.extend(list(self.arm2_encoder_pos_enc))
        if self.use_av_arm:
            encoder_in_pos_embed.extend(list(self.av_encoder_pos_enc))

        cls_token_arm1 = self.cls_token_arm1.unsqueeze(0).repeat(batch_size, 1, 1)
        cls_token_arm2 = self.cls_token_arm2.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.use_av_arm:
            cls_token_av = self.cls_token_av.unsqueeze(0).repeat(batch_size, 1, 1)

        encoder_in_tokens = []

        # Robot state token.
        if self.use_robot_state:
            encoder_in_tokens.append(torch.cat([cls_token_arm1, arm1_state_proj], dim=1))
            encoder_in_tokens.append(torch.cat([cls_token_arm2, arm2_state_proj], dim=1))
            if self.use_av_arm:
                encoder_in_tokens.append(torch.cat([cls_token_av, av_state_proj], dim=1))


        # Camera observation features and positional embeddings.
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []
            cls_token_image = self.cls_token_image.unsqueeze(0).repeat(batch_size, 1, 1)
            # all_cam_pos_embeds.append()

            for cam_index in range(batch["observation.images"].shape[-4]):
                cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                # buffer
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension,
            # and move to (sequence, batch, dim).
            all_cam_features = torch.cat(all_cam_features, axis=-1)
            encoder_in_tokens.append(
                torch.cat([cls_token_image, einops.rearrange(all_cam_features, "b c h w -> b (h w) c")], dim=1)
                           )
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
            encoder_in_pos_embed.append(
                torch.cat([list(self.image_encoder_pos_enc)[0], list(einops.rearrange(all_cam_pos_embeds, "b c h w -> b (h w) c"))[0]], dim=0)
            )

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.cat(encoder_in_tokens, axis=1)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, axis=0)
        encoder_in_cls_pos_embed = torch.cat(list(self.cls_encoder_pos_enc))

        encoder_in_pos_embed = encoder_in_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)
        encoder_in_cls_pos_embed = encoder_in_cls_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)    

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed, pos_embed_cls=encoder_in_cls_pos_embed)

        # Prepare the decoder input.
        encoder_out_arm1 = encoder_out[:self.num_cls_tokens_arm + 7]
        encoder_in_pos_embed_arm1 = encoder_in_pos_embed[:self.num_cls_tokens_arm + 7]
        encoder_out_arm2 = encoder_out[self.num_cls_tokens_arm + 7:2*self.num_cls_tokens_arm + 14]
        encoder_in_pos_embed_arm2 = encoder_in_pos_embed[self.num_cls_tokens_arm + 7:2*self.num_cls_tokens_arm + 14]
        if self.use_av_arm:
            encoder_out_av = encoder_out[2*self.num_cls_tokens_arm + 14:3*self.num_cls_tokens_arm + 21]
            encoder_in_pos_embed_av = encoder_in_pos_embed[2*self.num_cls_tokens_arm + 14:3*self.num_cls_tokens_arm + 21]
            encoder_out_image = encoder_out[3*self.num_cls_tokens_arm + 21:]
            encoder_in_pos_embed_image = encoder_in_pos_embed[3*self.num_cls_tokens_arm + 21:]

            encoder_out_real = torch.cat([encoder_out_arm1[:self.num_cls_tokens_arm], encoder_out_arm2[:self.num_cls_tokens_arm], encoder_out_av[:self.num_cls_tokens_arm], encoder_out_image[self.num_cls_tokens_image:]], dim=0)
            encoder_in_pos_embed_real = torch.cat([encoder_in_pos_embed_arm1[:self.num_cls_tokens_arm], encoder_in_pos_embed_arm2[:self.num_cls_tokens_arm], encoder_in_pos_embed_av[:self.num_cls_tokens_arm], encoder_in_pos_embed_image[self.num_cls_tokens_image:]], dim=0)
        else:
            encoder_out_image = encoder_out[2*self.num_cls_tokens_arm + 14:]
            encoder_in_pos_embed_image = encoder_in_pos_embed[2*self.num_cls_tokens_arm + 14:]

            encoder_out_real = torch.cat([encoder_out_arm1[:self.num_cls_tokens_arm], encoder_out_arm2[:self.num_cls_tokens_arm], encoder_out_image[self.num_cls_tokens_image:]], dim=0)
            encoder_in_pos_embed_real = torch.cat([encoder_in_pos_embed_arm1[:self.num_cls_tokens_arm], encoder_in_pos_embed_arm2[:self.num_cls_tokens_arm], encoder_in_pos_embed_image[self.num_cls_tokens_image:]], dim=0)

        # encoder_out_real = torch.cat([encoder_out[:self.num_cls_tokens_arm], encoder_out[self.num_cls_tokens_arm+7:2*self.num_cls_tokens_arm+7], encoder_out[2*self.num_cls_tokens_arm+self.num_cls_tokens_image+14:]], dim=0)
        # encoder_in_pos_embed_real = torch.cat([encoder_in_pos_embed[:self.num_cls_tokens_arm], encoder_in_pos_embed[self.num_cls_tokens_arm+7:2*self.num_cls_tokens_arm+7], encoder_in_pos_embed[2*self.num_cls_tokens_arm+self.num_cls_tokens_image+14:]], dim=0)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out_real,
            encoder_pos_embed=encoder_in_pos_embed_real,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions


# class ACTEncoder(nn.Module):
#     """Convenience module for running multiple encoder layers, maybe followed by normalization."""

#     def __init__(self, config: InterACTConfig, is_vae_encoder: bool = False):
#         super().__init__()
#         self.is_vae_encoder = is_vae_encoder
#         num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
#         self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

#     def forward(
#         self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
#     ) -> Tensor:
#         for layer in self.layers:
#             x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
#         x = self.norm(x)
#         return x

class ACTEncoderLayer(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class InterACTEncoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()

        self.num_blocks = config.num_blocks

        self.segment_wise_encoder = nn.ModuleList([
            ACTEncoderLayer(config) for _ in range(config.num_blocks)
        ])

        self.cross_segment_encoder = nn.ModuleList([
            ACTEncoderLayer(config) for _ in range(config.num_blocks)
        ])

        
        self.arm_cls = config.num_cls_tokens_arm
        self.cam_cls = config.num_cls_tokens_image
        self.use_av_arm = config.use_av_arm

    def forward(self, segments, pos_embed, pos_embed_cls):
        segments = einops.rearrange(segments, "b s d -> s b d")

        segment_arm1 = segments[:self.arm_cls+7]
        segment_arm2 = segments[self.arm_cls+7:2*self.arm_cls+14]
        pos_embed_arm1 = pos_embed[:self.arm_cls+7]
        pos_embed_arm2 = pos_embed[self.arm_cls+7:2*self.arm_cls+14]
        if self.use_av_arm:
            segment_av = segments[2*self.arm_cls+14:3*self.arm_cls+21]
            segment_image = segments[3*self.arm_cls+21:]
            pos_embed_av = pos_embed[2*self.arm_cls+14:3*self.arm_cls+21]
            pos_embed_image = pos_embed[3*self.arm_cls+21:]
        else:
            segment_image = segments[2*self.arm_cls+14:]
            pos_embed_image = pos_embed[2*self.arm_cls+14:]

        
            
        for i in range(self.num_blocks):
            updated_segment_arm1 = self.segment_wise_encoder[i](segment_arm1, pos_embed_arm1)
            updated_segment_arm2 = self.segment_wise_encoder[i](segment_arm2, pos_embed_arm2)
            if self.use_av_arm:
                updated_segment_av = self.segment_wise_encoder[i](segment_av, pos_embed_av)
            updated_segment_image = self.segment_wise_encoder[i](segment_image, pos_embed_image)

            if self.use_av_arm:
                updated_cls_tokens = self.cross_segment_encoder[i](
                    torch.cat([
                        updated_segment_arm1[:self.arm_cls], 
                        updated_segment_arm2[:self.arm_cls], 
                        updated_segment_av[:self.arm_cls], 
                        updated_segment_image[:self.cam_cls]
                        ], dim=0), pos_embed_cls
                    )
            else:
                updated_cls_tokens = self.cross_segment_encoder[i](
                    torch.cat([
                        updated_segment_arm1[:self.arm_cls], 
                        updated_segment_arm2[:self.arm_cls], 
                        updated_segment_image[:self.cam_cls]
                        ], dim=0), pos_embed_cls
                    )
                
            segment_arm1 = torch.cat([updated_cls_tokens[:self.arm_cls], updated_segment_arm1[self.arm_cls:]], dim=0)
            segment_arm2 = torch.cat([updated_cls_tokens[self.arm_cls:2*self.arm_cls], updated_segment_arm2[self.arm_cls:]], dim=0)
            if self.use_av_arm:
                segment_av = torch.cat([updated_cls_tokens[2*self.arm_cls:3*self.arm_cls], updated_segment_av[self.arm_cls:]], dim=0)
                segment_image = torch.cat([updated_cls_tokens[3*self.arm_cls:], updated_segment_image[self.cam_cls:]], dim=0)
            else:
                segment_image = torch.cat([updated_cls_tokens[2*self.arm_cls:], updated_segment_image[self.cam_cls:]], dim=0)
        
        if self.use_av_arm:
            segments = torch.cat([segment_arm1, segment_arm2, segment_av, segment_image], dim=0)
        else:
            segments = torch.cat([segment_arm1, segment_arm2, segment_image], dim=0)
        
        return segments

class ACTDecoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


"""
Use separate encoder for vision
Use only cls tokens in decoder
use only image features in decoder
"""