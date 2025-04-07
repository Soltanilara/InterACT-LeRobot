# InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation

Official implementation of "InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation" (CoRL 2024).

## Overview

INTERACT is a vision-based bimanual imitation learning approach that uses hierarchical attention mechanisms to capture inter-dependencies between dual-arm joint states and visual inputs. The model features a Hierarchical Attention Encoder and a Multi-arm Decoder, enabling accurate control for complex bimanual manipulation tasks.

Check out details of this work on our [project page](https://soltanilara.github.io/interact/) and [paper](https://www.arxiv.org/abs/2409.07914).
We implemented InterACT code on [LeRobot](https://huggingface.co/lerobot) codebase for ease of use.

## Installation

Create a new conda environment
```bash
conda create -n interact python=3.10 -y
conda activate interact
```

```bash
pip install -e .
pip install -e ".[aloha]"
```

## Quick Start

### Training a Model

```python
python lerobot/scripts/train.py \
    policy=interact \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human
```

Default config can be found in `lerobot/configs/policy/interact.yaml`

## Model Architecture

INTERACT features:
- Hierarchical Attention Encoder: Processes multi-modal inputs through segment-wise and cross-segment attention mechanisms
- Multi-arm Decoder: Generates each arm's action predictions in parallel while sharing information between arms

## Citation

If you find this code useful, please cite:
```bibtex
@article{lee2024interact,
    title={InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation},
    author={Lee, Andrew and Chuang, Ian and Chen, Ling-Yuan and Soltani, Iman},
    journal={arXiv preprint arXiv:2409.07914},
    year={2024}
}

@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```