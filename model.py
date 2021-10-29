import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D, Activation, ZeroPadding2D, concatenate

image_shape = (256, 256, 3)
text_encod_shape = (4, 4, 300)

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