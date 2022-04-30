# Multi-task Learning with Attention for End-to-end Autonomous Driving
Test code for the CVPR 2021 Workshop on Autonomous Driving paper [Multi-task Learning with Attention for End-to-end Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2021W/WAD/html/Ishihara_Multi-Task_Learning_With_Attention_for_End-to-End_Autonomous_Driving_CVPRW_2021_paper.html)


**Dependencies**  
The scripts run on the dependencies listed below:
- Carla 0.8.4 (using docker in our case)
- Docker: 20.10.5 (NVIDIA Docker: 2.5.0)

Hardware that we run our experiments on:
- Graphics card (GeForce RTX3080 level is recommended at least)
- Ubuntu 20.04

See more info on hardware requirements for CARLA: https://carla.readthedocs.io/en/0.8.4/faq/  
And here is another docs for runnning in a docker: https://carla.readthedocs.io/en/0.9.6/carla_docker/#nvidia-docker2


## Installation
The easiest way to build the env is to use pyenv and venv or other env-handler.
The code provided is tested on Python 3.6.8.
Note: if you go with pyenv, 1.2.24 is recommended: https://github.com/pyenv/pyenv/releases/tag/1.2.24
```bash
$ git clone https://github.com/KeishiIshihara/multitask-with-attention.git --recursive
$ cd multitask-with-attention
$ pyenv install 3.6.8
$ pyenv shell 3.6.8
$ python -m venv env
$ source env/bin/activate
$ (env) pip install --upgrade pip setuptools
$ (env) pip install -r requirements.txt
```

## Quick Start
To run our pretrained agent to see how it drives:
```bash
# Download weights from archive
$ ./download_weights.sh
# Start Carla server with docker
$ docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4 /bin/bash CarlaUE4.sh /Game/Maps/Town01 --world-port=2000 -benchmark -fps=10
# Activate env if you haven't
$ source env/bin/activate
# Run a pretrained agent in another console
(env) $ python enjoy.py --agent MTA --city-name Town01 --port 2000
```
You should now be able to see the video rendered from a third-person perspective with carera rgb input and action values that the agent is taking.
Note that this scenario is not from the carla benchmark test, but only some basic episodes.


## GradCam heatmaps
Once you have downloaded pretrained weights, you can generate gradcam heatmaps on your own. The command below generates heatmaps to `/reports/saliency_maps` by default:
```
$ python gradcam.py --h5-dataset-path /path/to/your/dataset.h5
```
For more options, see [gradcam.py](./gradcam.py)


## Troubleshooting
Please refer to [troubleshooting.md](./docs/troubleshooting.md).

## Citation
If you find this repo to be useful in your research, please consider citing our work.
```
@InProceedings{Ishihara_2021_CVPR,
    author    = {Ishihara, Keishi and Kanervisto, Anssi and Miura, Jun and Hautamaki, Ville},
    title     = {Multi-Task Learning With Attention for End-to-End Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {2902-2911}
}
```
