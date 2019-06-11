# Custom Gibson Environments for MID-LEVEL VISUAL REPRESENTATIONS FOR ACTIVE TASKS

In this repository we provide the custom environments we used to train models in our paper, **Mid Level Visual Features Improve Sample-Efficiency and Generalization for Active Tasks**. 

## Environments Available

### Visual Navigation

<img src=misc/visual_navigation.gif width=512>

The agent must navigate to a wooden cube. It receives only RGB frames as input.

### Visual Exploration

<img src=misc/visual_exploration.gif width=512>

The agent is equipped with a myopic laser sensor and is tasked with exploring as much as possible. The space is divided into a grid of voxels and the agent receives reward for uncovering as many squares as possible. The agent receives RGB frames and its current occupancy grid as input.

### Local Planning

<img src=misc/local_planning.gif width="512">

The agent is given RGB frames and the vector to a nonvisual target. It is tasked with navigating along the target vector to the goal, while being penalized for hitting obstacles. 
 

### Paper
**["Mid Level Visual Features Improve Sample-Efficiency and Generalization for Active Tasks"](http://perceptual/actor)**
**["Gibson Env: Real-World Perception for Embodied Agents"](http://gibson.vision/)**, in **CVPR 2018 [Spotlight Oral]**.

### Installation

#### System requirements

Installation via Docker is the preferred method. You need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. For source installation, see the main Gibson repository. The minimum system requirements are the following:

For docker installation: 
- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation.

To build your own docker image:

```bash
git clone https://github.com/bradleyemi/gibson.git
cd gibson
docker build . -t gibson
```

### Quick Start

See `gibson/examples/demo/play_husky_camera.py` for an example of how to interface with our OpenAI gym-like environment, and `gibson/examples/demo/train_husky_navigate_ppo2` for a training example.

### Further Customization

In `gibson/examples/config` you can view and modify the example config files to change the environment parameters.
