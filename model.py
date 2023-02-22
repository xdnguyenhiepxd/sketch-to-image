import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display


OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def res_block(filters, size,strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    return result
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# Build the generator
def Generator():
    inputs = tf.keras.layers.Input(shape=[1024, 1024, 3])

    down_stack = [
        downsample(32, 3,apply_batchnorm=False),  #return (bs, 512, 512, 64)
        downsample(64, 3),#256
        downsample(128, 3),#128
        downsample(256, 3),#64
        downsample(256, 3),#32
        downsample(256, 3),#16
    
    ]
    res = [
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3),
        res_block(256, 3)
    ]
       
    
        # # res_block(512, 3),
        # # res_block(512, 3),
        # # res_block(512, 3),
        # # res_block(512, 3),
        # # res_block(512, 3),
        # # res_block(512, 3),
        # # res_block(512, 3)
    # ]
    up_stack = [
        upsample(256, 3, apply_dropout=True), # (bs, 2, 2, 1024)
        # upsample(1024, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        # upsample(1024, 4), # (bs, 16, 16, 1024)
        upsample(256, 3), # (bs, 32, 32, 1024)
   
        upsample(128, 3),
        # (bs, 64, 64, 1024)      # (bs, 128, 128, 512)
        upsample(64, 3),       # (bs, 128, 128, 512)
        upsample(32, 3)
       
      # (bs, 256, 256, 256)
        #upsample(64, 4), # (bs, 512, 512, 128)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        #print(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    #x_down_end = x
    for r in res:
        #x = tf.keras.layers.Concatenate()([x, x_down_end])
        #print(x)
        res = r(x)
        x = x + res
        x = tf.keras.layers.ReLU()(x)
       
    #x = tf.keras.layers.Concatenate()([x, x_down_end])
    # print(x)
    # import time
    # time.sleep(500)
    # for up in up_stack:
        
        # x = up(x)
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x) 
        #print(x)
        #print("aaaaaaaaaaaaaaaaa")
        #print(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    # import time
    # time.sleep(500)
   
    return tf.keras.Model(inputs=inputs, outputs=x)


# Build the discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[1024, 1024, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[1024, 1024, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 1024, 1024, channels*2)

    down1 = downsample(32, 3, False)(x) # (bs, 512, 512, 64)
    down2 = downsample(64, 3)(down1) # (bs, 256, 256, 128)
    down3 = downsample(128, 3)(down2) # (bs, 128, 128, 256)
    #down4 = downsample(256, 3)(down3) # (bs, 64, 64, 256)
    #down5 = downsample(256, 3)(down4) # (bs, 32, 32, 256)


    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(256, 3, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

from tensorflow.keras.applications.vgg16 import VGG16
def build_vgg16():
    # load pre-trained VGG16 model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # get output layers for style and content layers
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    #content_layers = ['block5_conv2']
    #outputs = [model.get_layer(layer).output for layer in style_layers + content_layers]
    outputs = [model.get_layer(layer).output for layer in style_layers]
    # build model
    model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    return model
