# midnight struggle hrs can I get an amen

# 2021-Text-to-Image-Generation
# Description
This project is our own implementataion of text-to-image generation for birds. Based off of descirpiton provided by the user, it tries to create its own bird. It runs off of Python 3 and uses GAN to train data.
# Video Demonstration

# Colab Test and Visualization

# Installing the Environment
After cloning the github, change directory into it. If using anaconda, run command 
```
conda env create environment.yml
conda acivate text-to-image-generation-env
``` 
into the console to activate the yaml file with all the required libraries. Additionally, install `gdown` in the comand prompt after activating the environment 
# Directory Guide
birds


# Obtaining and Training Dataset
To begin training with the dataset we used, run `python model.py`. It checks if the CUB_200_2011 directory that contains the dataset exists, and if not, downloads and sets up the directory for it. The model trains the data, and saves them in the current directory.
# Testing Model

# References
