# Neural Supersampling

This is a work-in-progress unofficial re-implementation of the real-time neural supersampling model proposed in `Neural supersampling for real-time rendering` [[`Paper`](https://dl.acm.org/doi/10.1145/3386569.3392376)] using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). This is in no way endorsed by the original authors.

## Differences
This model is implemented as closely to the original paper as possible. However, there are some important differences:

- the original training data is not freely available. Therefore [Blender](https://www.blender.org/) is used to render images with color, depth and motion data from [Blender Open Movies](https://studio.blender.org/films/).
- the original paper seems to use motion data of the target resolution. Here, due to storage constraints, we use motion data of the source resolution
- the original paper seems to use raw depth values for feature extraction. I found high depth values to negatively impact numerical stability and therefore decided to use inverse depth, i.e. disparity, instead.

## Rendering
The training data may be rendered by Blender and the Cycles rendering engine. To achieve this, download any number of [Blender Open Movie](https://studio.blender.org/films/) assets and configure them in [render_all.py](rendering/render_all.py). Then either run [render_all.py](rendering/render_all.py) directly or use [run_blender_headless.sh](rendering/run_blender_headless.sh) to run Blender via Docker.

## Training
The training, evaluation and visualization are all implemented as separate files in the [model](model) directory. Alternatively, take a look at the Jupyter Notebook [NeuralSupersampling.ipynb](NeuralSupersampling.ipynb) <a href="https://colab.research.google.com/github/timmh/neural-supersampling/blob/main/NeuralSupersampling.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

## TODO
- [ ] train to convergence
- [ ] optimize using [TensorRT](https://github.com/pytorch/TensorRT) and embed in real-time application