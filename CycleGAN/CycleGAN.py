import os
import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from PIL import Image
import glob

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

def train(
    content_img_path,
    style_img_path,
    checkpoint_filepath,
    output_folder,
    epoch_num = 40,
    buffer_size = 256,
    batch_size = 1,
):
    tfds.disable_progress_bar()
    autotune = tf.data.AUTOTUNE
    ####################################
    #Prepare training dataset
    ####################################
    trainA_imgs = []
    trainA_label = []
    for filename in glob.glob(content_img_path):
        rgba_image = Image.open(filename)
        rgb_image = rgba_image.convert('RGB')
        tensor_img = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
        label = tf.constant(0, dtype=tf.int64)
        trainA_imgs.append(tensor_img)
        trainA_label.append(label)

    trainA_size = min(1000, len(trainA_imgs))
    trainA_imgs = random.sample(trainA_imgs, trainA_size)
    trainA_label = trainA_label[:trainA_size]
    trainA = tf.data.Dataset.from_tensor_slices((trainA_imgs, trainA_label))

    trainB_imgs = []
    trainB_label = []
    for filename in glob.glob(style_img_path):
        rgba_image = Image.open(filename)
        rgb_image = rgba_image.convert('RGB')
        tensor_img = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
        label = tf.constant(1, dtype=tf.int64)
        trainB_imgs.append(tensor_img)
        trainB_label.append(label)

    trainB_size = min(1000, len(trainB_imgs))
    trainB_imgs = random.sample(trainB_imgs, trainB_size)
    trainB_label = trainB_label[:trainB_size]
    trainB = tf.data.Dataset.from_tensor_slices((trainB_imgs, trainB_label))

    ####################################
    # Create training "Dataset" object
    ####################################
    # Apply the preprocessing operations to the training data
    trainA = (
        trainA.map(preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )
    trainB = (
        trainB.map(preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )
    ####################################
    # Visualize some samples
    ####################################
    _, ax = plt.subplots(5, 2, figsize=(10, 15))
    for i, samples in enumerate(zip(trainA.take(5), trainB.take(5))):
        classA = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
        classB = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
        ax[i, 0].imshow(classA)
        ax[i, 1].imshow(classB)
    plt.show()

    # prepare the inception v3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    # Callbacks
    plotter = GANMonitor(inception_model, trainA, output_folder)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True
    )

    cycle_gan_model = get_model()
    # Here we will train the model for just one epoch as each epoch takes around
    # 7 minutes on a single P100 backed machine.
    cycle_gan_model.fit(
        tf.data.Dataset.zip((trainA, trainB)),
        epochs=epoch_num,
        callbacks=[plotter, model_checkpoint_callback]
    )
    weight_file_path = checkpoint_filepath.replace("{epoch}", str(epoch_num))
    return weight_file_path

def test(
    input_img_path,
    weight_file_path,
    output_folder,
    buffer_size = 256,
    batch_size = 1,
):
    tfds.disable_progress_bar()
    autotune = tf.data.AUTOTUNE
    ####################################
    # Prepare testing dataset
    ####################################
    testA_imgs = []
    testA_label = []
    for filename in glob.glob(input_img_path):
        rgba_image = Image.open(filename)
        rgb_image = rgba_image.convert('RGB')
        tensor_img = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
        label = tf.constant(0, dtype=tf.int64)
        testA_imgs.append(tensor_img)
        testA_label.append(label)

    testA_size = len(testA_imgs)
    testA = tf.data.Dataset.from_tensor_slices((testA_imgs, testA_label))
    ####################################
    # Create testing "Dataset" object
    ####################################
    # Apply the preprocessing operations to the test data
    testA = (
        testA.map(preprocess_test_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )

    # prepare the inception v3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    cycle_gan_model = get_model()

    ####################################
    # Load the weights
    ####################################
    cycle_gan_model.load_weights(weight_file_path).expect_partial()
    print("Weights loaded successfully")

    imgs = []
    predictions = []
    _, ax = plt.subplots(1, 2, figsize=(10, 15))
    for i, img in enumerate(testA.take(testA_size)):
        prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
        ax[0].imshow(img)
        ax[1].imshow(prediction)
        ax[0].set_title("Input image")
        ax[1].set_title("Translated image")
        ax[0].axis("off")
        ax[1].axis("off")
        imgs.append(img)
        predictions.append(prediction)

        prediction = keras.preprocessing.image.array_to_img(prediction)
        prediction.save(output_folder+"predicted_img_{i}.png".format(i=i))

        plt.tight_layout()
        plt.savefig(output_folder+"results_img_{i}.png".format(i=i))
    plt.clf()
    plt.cla()
    plt.close()

    imgs = np.asarray(imgs)
    predictions = np.asarray(predictions)
    # convert integer to floating point values
    imgs = imgs.astype('float32')
    predictions = predictions.astype('float32')
    # resize images
    imgs = scale_images(imgs, (299, 299, 3))
    predictions = scale_images(predictions, (299, 299, 3))
    # pre-process images
    imgs = preprocess_input(imgs)
    predictions = preprocess_input(predictions)
    # calculate fid
    fid = calculate_fid(inception_model, imgs, predictions)
    print(f"FID: {fid}")



def get_model():
    ####################################
    # Hyper parameters
    ####################################
    # Weights initializer for the layers.
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # Gamma initializer for instance normalization.
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # Get the generators
    gen_G = get_resnet_generator(kernel_initializer=kernel_init, gamma_initializer=gamma_init, name="generator_G")
    gen_F = get_resnet_generator(kernel_initializer=kernel_init, gamma_initializer=gamma_init, name="generator_F")

    # Get the discriminators
    disc_X = get_discriminator(kernel_initializer=kernel_init, gamma_initializer=gamma_init, name="discriminator_X")
    disc_Y = get_discriminator(kernel_initializer=kernel_init, gamma_initializer=gamma_init, name="discriminator_Y")

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    return cycle_gan_model

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, (72, 72))
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=(64, 64, 3))
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, (64, 64))
    img = normalize_img(img)
    return img

####################################
#Building blocks used in the CycleGAN generators and discriminators
####################################

class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

####################################
#Build the generators
####################################

def get_resnet_generator(
    kernel_initializer,
    gamma_initializer,
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    name=None,
):
    img_input = layers.Input(shape=(64, 64, 3), name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_initializer, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"), kernel_initializer=kernel_initializer, gamma_initializer=gamma_initializer)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"), kernel_initializer=kernel_initializer, gamma_initializer=gamma_initializer)

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"), kernel_initializer=kernel_initializer, gamma_initializer=gamma_initializer)

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

####################################
#Build the discriminators
####################################

def get_discriminator(
    kernel_initializer,
    gamma_initializer,
    filters=64,
    num_downsampling=3,
    name=None
):
    img_input = layers.Input(shape=(64, 64, 3), name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                kernel_initializer = kernel_initializer,
                gamma_initializer = gamma_initializer
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
                kernel_initializer=kernel_initializer,
                gamma_initializer=gamma_initializer
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model

####################################
#Build the CycleGAN model
####################################

class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(self, batch_data):
        # x is class A and y is class B
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # class A to fake class B
            fake_y = self.gen_G(real_x, training=True)
            # class B to fake class A -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (class A to fake class B to fake class A): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (class B to fake class A to fake class B) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

####################################
# Calculate FID
####################################

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

####################################
# Create a callback that periodically saves generated images
# and illustrate line chart of losses vs epoch_num
####################################

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, inception_model, load_data_fn, output_folder, num_img=4):
        self.num_img = num_img
        self.inception_model = inception_model
        self.load_data_fn = load_data_fn
        self.output_folder = output_folder
        self.total_loss_G = []
        self.total_loss_F = []
        self.disc_A_loss = []
        self.disc_B_loss = []
        self.epoch_num = []
        self.fid_scores = []
        self.imgs = []
        self.predictions = []

    def on_epoch_end(self, epoch, logs=None):
        for i, img in enumerate(self.load_data_fn.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            self.imgs.append(img)
            self.predictions.append(prediction)

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                self.output_folder+"generated_img_{epoch}_{i}.png".format(i=i, epoch=epoch + 1)
            )

        # plot generator losses
        self.epoch_num.append(epoch+1)
        self.total_loss_G.append(logs['G_loss'])
        self.total_loss_F.append(logs['F_loss'])
        if (epoch+1) % 10 == 0:
            plt.plot(self.epoch_num, self.total_loss_G, label='total_generator_loss_G')
            plt.plot(self.epoch_num, self.total_loss_F, label='total_generator_loss_F')
            plt.title('CycleGan generator losses VS #epoch')
            plt.xlabel('#epoch')
            plt.ylabel('generator losses')
            plt.legend()
            plt.savefig(self.output_folder+"generator_losses_{epoch}.png".format(epoch=epoch+1))
            plt.clf()
            plt.cla()
            plt.close()

        # plot discriminator losses
        self.disc_A_loss.append(logs['D_X_loss'])
        self.disc_B_loss.append(logs['D_Y_loss'])
        if (epoch+1) % 10 == 0:
            plt.plot(self.epoch_num, self.disc_A_loss, label='total_discriminator_loss_A')
            plt.plot(self.epoch_num, self.disc_B_loss, label='total_discriminator_loss_B')
            plt.title('CycleGan discriminator losses VS #epoch')
            plt.xlabel('#epoch')
            plt.ylabel('discriminator losses')
            plt.legend()
            plt.savefig(self.output_folder+"discriminator_losses_{epoch}.png".format(epoch=epoch+1))
            plt.clf()
            plt.cla()
            plt.close()

        imgs = np.asarray(self.imgs)
        predictions = np.asarray(self.predictions)
        # convert integer to floating point values
        imgs = imgs.astype('float32')
        predictions = predictions.astype('float32')
        # resize images
        imgs = scale_images(imgs, (299, 299, 3))
        predictions = scale_images(predictions, (299, 299, 3))
        # pre-process images
        imgs = preprocess_input(imgs)
        predictions = preprocess_input(predictions)
        # calculate fid
        fid = calculate_fid(self.inception_model, imgs, predictions)
        self.fid_scores.append(fid)
        # plt.show()
        if (epoch+1) % 10 == 0:
            # plot fid scores vs #epoch
            plt.plot(self.epoch_num, self.fid_scores)
            plt.title('FID scores VS #epoch')
            plt.xlabel('#epoch')
            plt.ylabel('FID scores')
            plt.savefig(self.output_folder+"FID_scores_{epoch}.png".format(epoch=epoch+1))
            plt.clf()
            plt.cla()
            plt.close()

####################################
#Train the end-to-end model
####################################
# Define the loss function for the generators
def generator_loss_fn(fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = keras.losses.MeanSquaredError()
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = keras.losses.MeanSquaredError()
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


