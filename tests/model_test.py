import pytest
from model import build_discriminator, build_generator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from matplotlib import image

def test_dis_output_shape():
	dis = build_discriminator()
	dis_optimizer = Adam(lr=0.1, beta_1=0.5, beta_2=0.999)
	dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
	im = image.imread("Yoda.jpg")
	prediction = dis([tf.expand_dims(im, 0),  tf.expand_dims(np.empty(shape=(4,4,300),dtype='int'), 0)])
	assert(prediction.numpy().shape == (1,1))

def test_dis_output_value():
	dis = build_discriminator()
	dis_optimizer = Adam(lr=0.1, beta_1=0.5, beta_2=0.999)
	dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
	im = image.imread("Yoda.jpg")
	prediction = dis([tf.expand_dims(im, 0),  tf.expand_dims(np.empty(shape=(4,4,300),dtype='int'), 0)])
	assert(prediction.numpy()[0][0] >= 0 and prediction.numpy()[0][0] <= 1)

def test_gen_output_shape():
	gen = build_generator()
	gen_optimizer = Adam(lr= 0.0002, beta_1=0.5, beta_2=0.999)
	gen.compile(loss="binary_crossentropy", optimizer=gen_optimizer)
	x, mean_logsigma = gen([tf.expand_dims(np.empty(shape=(4800,),dtype='int'), 0), tf.expand_dims(np.empty(shape=(100,),dtype='int'), 0)])
	assert(x.numpy().shape == (1,256,256,3))