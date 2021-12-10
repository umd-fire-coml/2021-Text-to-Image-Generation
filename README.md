# 2021-Text-to-Image-Generation
# Description
This project is our own implementataion of text-to-image generation for birds. Based off of descirpiton provided by the user, it tries to create an original bird image. It runs in Python 3 and uses a target-aware generative averserial model.
# Video Demonstration
youtube.com/watch?v=FgwgQJRBPAc
# Colab Notebook
https://colab.research.google.com/drive/1mfxWs4v8WekAs4snLyXvPenQz0g7b2ZY?usp=sharing
# Installing the Environment
After cloning the github, change directory into it. If using anaconda, run command 
```
conda env create environment.yml
conda acivate text-to-image-generation-env
``` 
into the console to activate the yaml file with all the required libraries. Additionally, install `gdown` in the comand prompt after activating the environment 
# Directory Guide
birds
```
Contains dataframes for filenames (filename.pickle) and info of classes (class_info.pickle) for each image for testing and training.
```
logs/train
```
Logs training progress over time.
```
environment_check.py
```
checks that the right versions of libraries from the environment were installed.
```
environment.yml
```
File use to install the required libraries.
```
test
```
Tests used during development.
```
requirements.txt
```
Required packages for code.
```
stage1_dis.h5
```
Checkpoint for discriminator after training.
```
stage1_gen.h5
```
Checkpoint for generator after training.
```
test-requirements.txt
```
Requirements for testing in git workflow
```

# Obtaining and Training Dataset
To begin training with the dataset we used, run `python model.py`. It checks if the CUB_200_2011 directory that contains the dataset exists, and if not, downloads and sets up the directory for it. The model trains the data, and saves them in the current directory.
# Testing Model

# References
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitrius Metaxas. StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks," arXiv, August 5, 2017. [Online]. Available: https://arxiv.org/abs/1612.03242v2. [Accessed December 06, 2021]
