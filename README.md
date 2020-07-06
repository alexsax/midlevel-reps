# `Mid-Level Visual Representations` Improve Generalization and Sample Efficiency for Learning Visuomotor Policies</h1>


What happens when robots leverage _visual priors_ during learning? They learn faster, generalize better, and achieve higher final performance.

### [Winner of CVPR19 Habitat Embodied Agents Challenge](https://ai.facebook.com/blog/open-sourcing-ai-habitat-an-simulation-platform-for-embodied-ai-research/)

<div align="center">
  <img src="https://github.com/alexsax/midlevel-reps/blob/helper/img/teaser.gif"  width="900px" />
</div>

**An agent with mid-level perception** navigating inside a building.

<br>

**Summary:** How much does having visual priors about the world (e.g. the fact that the world is 3D) assist in learning to perform downstream motor tasks (e.g. delivering a package)? We study this question by integrating a generic perceptual skill set (mid-level vision) within a reinforcement learning framework. This skill set provides a policy with a more processed state of the world compared to raw images, conferring significant advantages over training from scratch (i.e. not leveraging priors) in navigation-oriented tasks. Agents are able to generalize to situations where the from-scratch approach fails and training becomes significantly more sample efficient. Realizing these gains requires careful selection of the mid-level perceptual skills, and we provide an efficient and generic max-coverage feature set that can be adopted in lieu of raw images.

This repository includes [code](https://github.com/alexsax/midlevel-reps/tree/master/evkit) from the paper, [ready-made dockers](#running-our-experiments-) containing pre-built environments, and commands to [run our experiments](#step-3-run-the-experiment). We also include instructions to install the lightweight [`visualpriors` package](#using-mid-level-perception-in-your-code-), which allows you to use mid-level perception in your own code as a drop-in replacement for pixels.

Please see the website (http://perceptual.actor/) for more technical details. This repository is intended for distribution of the code, environments, and installation/running instructions.

#### See more mid-level perception results and then try mid-level perception out for yourself
| [Online demos](http://perceptual.actor/policy_explorer/) | [Run our examples](#running-our-experiments-) | [Try it yourself]() | 
|:----:|:----:|:----:|
| [![Online demos](https://github.com/alexsax/midlevel-reps/blob/helper/img/policy_explorer.png)](http://perceptual.actor/policy_explorer/) | [<img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/crazy_tb.png >](#running-our-experiments-) | Using ```visualpriors```! [![Try it yourself](https://github.com/alexsax/midlevel-reps/blob/helper/img/import_text.png)](#using-mid-level-perception-in-your-code-) |

<div align="center">
  <p>Overview Video (6 min)</p>
  <a href=https://www.youtube.com/watch?v=HtefpenfxTQ&feature=youtu.be><img src="https://github.com/alexsax/midlevel-reps/blob/helper/img/youtube_screenshot.png?raw=true"  width="400px" /></a>
</div>


#### Papers

[**Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies**,](https://arxiv.org/abs/1812.11971)<br>
_Arxiv 2018_.<br>
Alexander Sax, Bradley Emi, Amir Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.

[**Learning to Navigate Using Mid-Level Visual Priors**,](https://arxiv.org/pdf/1912.11121.pdf)<br>
_CoRL 2019_.<br>
Alexander Sax, Jeffrey O. Zhang, Bradley Emi, Amir Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.

<br>

## Contents 
- [Quickstart](#quickstart-)
- [Running experiments](#running-our-experiments-)
  - [Minimal docker with Habitat experiment](#experiments-in-Habitat)
  - [Docker with all environments (Gibson, Habitat, Doom)](#experiments-in-gibson-and-vizdoom-under-construction)
- [Using mid-level perception in your code](#using-mid-level-perception-in-your-code-)
  - [Installation (using pip)](#installing-visualpriors)
  - [How to use the `visualpriors` package](#using-visualpriors)
- [Embodied Vision Toolkit](#embodied-vision-toolkit-under-construction-)
- [Citation](#citation)

<br>

## Quickstart [\[^\]](#Contents)

Quickly transform an image into `surface normals` features and then visualize the result. 

**Step 1)** Run `pip install visualpriors` to install the `visualpriors` package. You'll need `pytorch`!

**Step 2)** Using python, download an image to `test.png` and visualize the readout in `test_normal_readout.png`

```python
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess

# Download a test image
subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

# Load image and rescale/resize to [-1,1] and 3x256x256
image = Image.open('test.png')
x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
x = x.unsqueeze_(0)

# Transform to normals feature
representation = visualpriors.representation_transform(x, 'normal', device='cpu')

# Transform to normals feature and then visualize the readout
pred = visualpriors.feature_readout(x, 'normal', device='cpu')

# Save it
TF.to_pil_image(pred[0] / 2. + 0.5).save('test_normals_readout.png')
```


| Input image | `representation` (3 of 8 channels) | `pred` (after readout) | 
|:----:|:----:|:----:|
| <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example1.png height=256 width=256> | <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example1__normal_features.png height=48 width=48> | <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example1__rgb2sfnorm.png height=256 width=256> |


In addition to normals, you can use any of the following features in your transform:
```
autoencoding          depth_euclidean          jigsaw                  reshading          
colorization          edge_occlusion           keypoints2d             room_layout      
curvature             edge_texture             keypoints3d             segment_unsup2d        
class_object          egomotion                nonfixated_pose         segment_unsup25d
class_scene           fixated_pose             normal                  segment_semantic      
denoising             inpainting               point_matching          vanishing_point
```
A description of each of the features is contained in the supplementary of [Taskonomy](http://taskonomy.vision).

<br>

## Running our experiments [\[^\]](#Contents)

Using mid-level vision, it is possible to train an agent in only a _single room_ and then generalize the training to novel spaces in different buildings. The feature-based agents learn faster and perform significantly better than their trained-from-scratch counterparts. For more extensive discussions about the benefits of visual priors and mid-level vision in particular, please see the [paper](http://perceptual.actor). This repository focuses on delivering easy-to-use experiments and code.

We provide dockers to reproduce and extend our results. Setting up these environments can be a pain, and docker provides a containerized environment with the environments already set up. If not already installed, install [Docker](https://docs.docker.com/install/) and [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker#quickstart).

![environments](https://github.com/alexsax/midlevel-reps/blob/helper/img/mesh_figure_short.png?raw=true)


<br>

### Experiments in [Habitat](https://github.com/facebookresearch/habitat-api)

In the [main paper](http://perceptual.actor/main_paper.pdf) we studied how mid-level perception affects learning on various tasks. In the `local planning` task, 
> The agent must direct itself to a given nonvisual target destination (specified using coordinates) using visual inputs, avoiding obstacles and walls as it navigates to the target. This task is useful for the practical skill of local planning, where an agent must traverse sparse waypoints along a desired path. The agent receives dense positive reward proportional to the progress it makes (in Euclidean distance) toward the goal.
Further details are contained in the paper. 

The following steps will guide you through training an agent to do the `local planning` task in the  [Habitat](https://github.com/facebookresearch/habitat-api) environment. The following agents were submitted to the [Habitat Challenge](https://aihabitat.org/workshop/)

![Habitat experiment](https://github.com/alexsax/midlevel-reps/blob/helper/img/habitat.gif?raw=true)

**An agent navigating to the goal.** The goal is shown in the middle panel, in green. The agent sees only the left and right panels. 


#### Step 1) Install the docker (22 GB)
In a shell, pull the docker to your local machine
``` bash
docker pull activeperception/habitat:1.0
```

#### Step 2) Start up a docker container
Once the docker is installed you can start a new container. The following command will start a new container that can use ports on the host (so that visdom can be run from within the container).
``` bash
docker run --runtime=nvidia -ti --rm \
    --network host --ipc=host \
    activeperception/habitat:1.0 bash
```

#### Step 3) Start visualization services
Inside the docker container we can start a visdom server (to view videos) and a tensorboard instance (for better charts).
```
mkdir /tmp/midlevel_logs/
screen -S visdom_server -p 0 -X stuff "visdom^M"
screen -S visdom_server -p 0 -X stuff "tensorboard --logdir .^M"
```

#### Step 3) Run the experiment
Lastly, we just need to start the experiment. Let's try training an agent that uses predicted **surface normals** as inputs. We'll use only 1 training and 1 val process since we're just trying to visualize the results. 
```
python -m scripts.train_rl /tmp/midlevel_logs/normals_agent run_training with uuid=normals cfg_habitat taskonomy_decoding  cfg.saving.log_interval=10 cfg.env.num_processes=2 cfg.env.num_val_processes=1
```

If you want to compare this to an agent trained from **scratch**, you can swap this easily with:
```
python -m scripts.train_rl /tmp/midlevel_logs/scratch run_training with uuid=scratch cfg_habitat scratch  cfg.saving.log_interval=10 cfg.env.num_processes=2 cfg.env.num_val_processes=1
```

Or a **blinded** agent (no visual input)
```
python -m scripts.train_rl /tmp/midlevel_logs/blind run_training with uuid=blind cfg_habitat blind  cfg.saving.log_interval=10 cfg.env.num_processes=2 cfg.env.num_val_processes=1
```

Or using the **Max-Coverage Min-Distance Featureset**
```
python -m scripts.train_rl /tmp/midlevel_logs/max_coverage run_training with uuid=blind cfg_habitat max_coverage_perception  cfg.saving.log_interval=10 cfg.env.num_processes=2 cfg.env.num_val_processes=1
```
**Note**: You might see some NaNs in the first iteration. Not to worry! This is probably because the first logging occurs before any episodes have finished.

You can explore more configuration options in `configs/habitat.py`! We used [SACRED](https://sacred.readthedocs.io/en/latest/) for managing experiments, so any of these experiments can be easily modified from the command line.

<br>

### Experiments in [Gibson](https://github.com/StanfordVL/GibsonEnv) and [VizDoom](https://github.com/mwydmuch/ViZDoom) (Under Construction!)

In addition to `local_planning` in Habitat, we implemented this and other tasks in Gibson and VizDoom, again finding the same phenomena (better generalization and sample efficiency). The new tasks are defined as follows:

> **Navigation to a Visual Target:** In this scenario the agent must locate a specific target object (Gibson: a wooden crate, Doom: a green torch) as fast as possible with only sparse rewards. Upon touching the target there is a large one-time positive reward and the episode ends. Otherwise there is a small penalty for living. The target looks the same between episodes although the location and orientation of both the agent and target are randomized. The agent must learn to identify the target during the course of training.

> **Visual Exploration:** The agent must visit as many new parts of the space as quickly as possible. The environment is partitioned into small occupancy cells which the agent "unlocks" by scanning with a myopic laser range scanner. This scanner reveals the area directly in front of the agent for up to 1.5 meters. The reward at each timestep is proportional to the number of newly revealed cells. 

Full details are contained in the [main paper](http://perceptual.actor/main_paper.pdf). The following section will guide you through training agents to use either mid-level vision or raw pixels to perform these tasks in Gibson and VizDoom. 

![Gibson experiment](https://github.com/alexsax/midlevel-reps/blob/helper/img/gibson_planning.gif?raw=true)

**Local planning using surface normal features in Gibson.** We also implemented other tasks; Visual-Target Navigation and Visual Exploration are included in the docker. 


![Doom experiment](https://github.com/alexsax/midlevel-reps/blob/helper/img/doom_nav.gif?raw=true)

**Visual navigation in Doom.** The agent must navigate to the `green_torch`. The docker includes implementions of Visual-Target Navigation and also Visual Exploration in VizDoom.

**Note:** Our original results (in a code dump form) are currently public via the docker `activeperception/midlevel-training:0.3`. We are currently working on a cleaner and more portable release.
 

<br>
<br>

## Using mid-level perception in your code [\[^\]](#Contents)

In addition to using our dockers, we provide a simple way to use mid-level vision in your code. We provide the lightweight `visualpriors` package which contains functions to upgrade your agent's state from pixels to mid-level features. The `visualpriors` package seeks to be a drop-in replacement for raw pixels. The remainder of this section focuses installation and usage. 

<br>

### Installing _visualpriors_

**The simplest way** to install the `visualpriors` package is via pip: 
```bash
pip install visualpriors
```

If you would prefer to have the source code, then you can clone this repo and install locally via:
```bash
git clone --single-branch --branch visualpriors git@github.com:alexsax/midlevel-reps.git
cd midlevel-reps
pip install -e .
```

<br>

### Using _visualpriors_

Once you've installed `visualpriors` you can immediately begin using mid-level vision. The transform is as easy as 

```representation = visualpriors.representation_transform(x, 'normal', device='cpu')```

**1) A complete script for `surface normals` transform**
```python
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess

feature_type = 'normal'

# Download a test image
subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

# Load image and rescale/resize to [-1,1] and 3x256x256
image = Image.open('test.png')
o_t = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
o_t = o_t.unsqueeze_(0)

# Transform to normals feature
representation = visualpriors.representation_transform(o_t, feature_type, device='cpu') # phi(o_t) in the diagram below

# Transform to normals feature and then visualize the readout
pred = visualpriors.feature_readout(o_t, feature_type, device='cpu')

# Save it
TF.to_pil_image(pred[0] / 2. + 0.5).save('test_{}_readout.png'.format(feature_type))
```

Which produces the following results:

| Input image (`o_t`) | `representation` (3 of 8 channels) | After readout (pred) | 
|:----:|:----:|:----:|
| <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example2.png height=256 width=256> | <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example1__normal_features.png height=48 width=48> | <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/example2__rgb2sfnorm.png height=256 width=256> |

|||
|:----|:----:|
| **Diagram of the above setup in an active framework.** The input image (`o_t`) gets encoded into `representation=\phi(o_t)` which is decoded into the prediction `pred`. In this example, we choose to make the encoder (`phi`) a [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf). | <img src=https://github.com/alexsax/midlevel-reps/blob/helper/img/transfer.png > | 

**2) Now let's try transforming the image into  `object classification` (ImageNet) features**, instead of surface normals:
```python
midlevel_feats = visualpriors.representation_transform(pre_transform_img, features='class_object')  # So easy!
```

**3) In addition to `normals` and `class_object`, you can use any of the following features in your transform:**
```
autoencoding          depth_euclidean          jigsaw                  reshading          
colorization          edge_occlusion           keypoints2d             room_layout      
curvature             edge_texture             keypoints3d             segment_unsup2d        
class_object          egomotion                nonfixated_pose         segment_unsup25d
class_scene           fixated_pose             normal                  segment_semantic      
denoising             inpainting               point_matching          vanishing_point
```
A description of each of the features is contained in the supplementary of [Taskonomy](http://taskonomy.vision).


**4) You can even use multiple features at once:**
```python
from midlevel import multi_representation_transform

midlevel_feats = multi_representation_transform(pre_transform_img, # should be 3x256x256. 
                                                features=['normal', 'depth', 'class_object'])  
action = policy(midlevel_feats). # midlevel_feats will be (len(features)*8, 16, 16)
```

**5)** The obvious next question is: _what's a good general-purpose choice of features?_ I'm glad that you asked! Our **Max-Coverage Min-Distance Featureset** proposes an answer, and those solver-found sets are implemented in the function `max_coverage_transform`. For example, if you can afford to use three features:
```python
from visualpriors import max_coverage_transform

midlevel_feats = max_coverage_transform(pre_transform_img, featureset_size=3)
action = policy(midlevel_feats)
```

<br>
<br>
<br>

## Embodied Vision Toolkit (Under Construction!) [\[^\]](#Contents)

In addition to providing the lightweight `visualpriors` package, we provide code for our full research platform, `evkit`. This platform includes utilities for handling visual transforms, flexibility with the choice of RL algprothm (including our off-policy variant of PPO with replay buffer), and tools for logging and visualization.

This section will contain an overview of `evkit`, which is currently available in the `evkit/` folder of this repository.

<br>
<br>
<br>

### Citation

If you find this repository or toolkit useful, then please cite:

    @inproceedings{midLevelReps2018,
     title={Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies.},
     author={Alexander Sax and Bradley Emi and Amir R. Zamir and Leonidas J. Guibas and Silvio Savarese and Jitendra Malik},
     year={2018},
    }
