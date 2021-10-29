import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, ReLU, UpSampling2D, Conv2D, Activation, ZeroPadding2D, concatenate, Lambda, Concatenate

image_shape = (256, 256, 3)
text_encod_shape = (4, 4, 300)

def generate_c(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]

    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean

    return c

def build_generator():
    input_layer = Input(shape=(4800,))
    x = Dense(256, kernel_initializer='random_normal')(input_layer)
    mean_logsigma = LeakyReLU(alpha=0.2)(x)

    c = Lambda(generate_c)(mean_logsigma)

    input_layer2 = Input(shape=(100,))

    gen_input = Concatenate(axis=1)([c, input_layer2])

    x = Dense(128 * 8 * 4 * 4, use_bias=False, kernel_initializer='random_normal')(gen_input)
    x = ReLU()(x)

    x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, kernel_size=3, padding="same", strides=1, use_bias=False, kernel_initializer='random_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, padding="same", strides=1, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(4, 4))(x)
    x = Conv2D(128, kernel_size=3, padding="same", strides=1, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(4, 4))(x)
    x = Conv2D(64, kernel_size=3, padding="same", strides=1, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, kernel_size=3, padding="same", strides=1, kernel_initializer='random_normal', use_bias=False)(x)
    x = Activation(activation='tanh')(x)

    gen = Model(inputs=[input_layer, input_layer2], outputs=[x, mean_logsigma])
    return gen

def build_discriminator():
    input_layer = Input(shape=image_shape)

    x = Conv2D(64, text_encod_shape[:2],
               padding='same', strides=2, kernel_initializer='random_normal',
               input_shape=image_shape, use_bias=False)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, text_encod_shape[:2], padding='same', strides=2, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, text_encod_shape[:2], padding='same', strides=4, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, text_encod_shape[:2], padding='same', strides=4, kernel_initializer='random_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    input_layer2 = Input(shape=text_encod_shape)

    merged_input = concatenate([x, input_layer2])

    x2 = Conv2D(64 * 8, kernel_size=1, kernel_initializer='random_normal',
                padding="same", strides=1)(merged_input)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1)(x2)
    x2 = Activation('sigmoid')(x2)

    discriminator = Model(inputs=[input_layer, input_layer2], outputs=[x2])
    return discriminator