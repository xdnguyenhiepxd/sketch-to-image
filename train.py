import tensorflow as tf
import os
import pathlib
import time
import datetime
from model import Generator, Discriminator, build_vgg16
from loss import generator_loss, discriminator_loss, vgg_loss
#from loss import generator_loss, discriminator_loss
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg16 import preprocess_input
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
checkpoint_path = "/home/www/data/data/saigonmusic/Dev_AI/Sketch2Image_v2/weights/checkpoint.index"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
generator = Generator()
discriminator = Discriminator()
vgg16 = build_vgg16()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
SIZE = 1024


#generator.load_weights(latest)
def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


# The data training set consist of 100 images
BUFFER_SIZE = 100
# The batch size of 1 
BATCH_SIZE = 1


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, SIZE, SIZE, 3])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def random_jitter(input_image, real_image):
    # Resizing to 1024X1024
    input_image, real_image = resize(input_image, real_image, SIZE, SIZE)

    # Random cropping
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, SIZE, SIZE)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def save_out(model, img_input,step):
    prediction = model(img_input, training=True)
    plt.figure(figsize=(10, 10))
    prediction = prediction[0]*0.5 + 0.5
    plt.imshow(prediction)
    plt.axis('off')
    plt.savefig("/home/www/data/data/saigonmusic/Dev_AI/Sketch2Image_v2/out_img_"+str(step)+".jpg" ,bbox_inches='tight', dpi = 60, pad_inches = 0, transparent = True)


def gen_image(inp_path, out_path):
    print("kkk")
    inp_tensor = cv2.imread(inp_path)
    height, width, _ = inp_tensor.shape
    inp_tensor = cv2.resize(inp_tensor, (SIZE, SIZE))
    inp_tensor = (inp_tensor/127.5) - 1
    print(inp_tensor)
    print(checkpoint)
    inps = np.array([inp_tensor])
    prediction = generator(inps, training=True)[0]
    prediction = np.array(prediction)
    prediction = cv2.resize(prediction, (width, height))
    cv2.imwrite(out_path, prediction)
    return prediction


# Build train
def train_step(input_image, target, step):


    #Build loss vgg16
    gen_output = generator(input_image, training=True)
    int_gen_vgg = tf.image.resize(gen_output,(224,224))
    int_tar_vgg = tf.image.resize(target,(224,224))
    out_gen_vgg = vgg16(int_gen_vgg)
    out_tar_vgg = vgg16(int_tar_vgg)
    ts_out_gen = tf.convert_to_tensor(out_gen_vgg[0])
    h,w,c = ts_out_gen.get_shape().as_list()[1:]
    loss_vgg = vgg_loss(out_gen_vgg[0],out_tar_vgg[0],h,w,c)
    
    #Build Loss Gan, Dis
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
    
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
 
    
    
        loss_total_gen = loss_vgg + gen_total_loss
        #loss_total_gen =gen_total_loss
        
        
        if (step) % 1000 == 0:
            with open("/home/www/data/data/saigonmusic/Dev_AI/Sketch2Image_v2/lost.txt","a") as fl:
                fl.write(f"======================================Hien Thi {step}================================\n")
                fl.write(f"\nGen Loss: {gen_total_loss}\n")
                fl.write(f"\nGen 01 Loss: {gen_gan_loss}\n")
                fl.write(f"\nGen L1 Loss: {gen_l1_loss}\n")
                fl.write(f"\nVGG Loss: {loss_vgg}\n")
                fl.write(f"\nDis Loss: {disc_loss}\n")
    
    generator_gradients = gen_tape.gradient(loss_total_gen,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))



def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    #checkpoint_path = "/home/www/data/data/saigonmusic/Dev_AI/Sketch2Image_v2/weights/checkpoint"
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            #generate_images(generator, example_input, example_target)
            save_out(generator,example_input,step)
            print(f"Step: {step//1000}k")
        train_step(input_image, target, step)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 1000 == 0:
            path = "/home/www/data/data/saigonmusic/Dev_AI/Sketch2Image_v2/weight/"+str(step)
            checkpoint_path = path + "/checkpoint"
            os.mkdir(path)
            generator.save_weights(checkpoint_path)
            # checkpoint.save(file_prefix=checkpoint_prefix)


# Build an input pipeline with tf.data
train_dataset = tf.data.Dataset.list_files(f'/home/www/data/data/saigonmusic/Dev_AI/Sketch2Landscape/train_{SIZE}_street/*.jpg')
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(f'/home/www/data/data/saigonmusic/Dev_AI/Sketch2Landscape/val_{SIZE}_street/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)



if __name__ == "__main__":
    # Train
    fit(train_dataset, test_dataset, steps=40000)