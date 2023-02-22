import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display


LAMBDA = 2000
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define the generator loss
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.math.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = 10*gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_fake_loss(disc_eg_output):
    disc_fake_loss = 80*loss_object(tf.zeros_like(disc_eg_output), disc_eg_output)
    return disc_fake_loss

# Define the discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
#nhan them trong so
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + 100*generated_loss

    return total_disc_loss


def vgg_loss(vgg_gen, vgg_tar,h,w,c):
    
    l1_loss = 1000*tf.math.reduce_mean(tf.abs(vgg_tar - vgg_gen))/(h*w*c)
    return l1_loss
