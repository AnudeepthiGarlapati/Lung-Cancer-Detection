import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2DTranspose

class VariationalAutoencoder:
    def __init__(self, image_size=(128, 128), latent_dim=100):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.vae = self.build_vae()

    def build_vae(self):
        # Encoder
        input_img = Input(shape=self.image_size + (3,))
        x = Conv2D(32, 3, activation='relu', padding='same')(input_img)
        x = Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)  # Downsample here
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape((shape_before_flattening[1], shape_before_flattening[2], shape_before_flattening[3]))(x)  # Ensure correct shape for subsequent layers
        x = Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)  # Upsample here
        decoder_output = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

        # Define VAE model
        z = Lambda(sampling)([z_mean, z_log_var])
        encoder = Model(input_img, z)
        decoder = Model(decoder_input, decoder_output)
        z_decoded = decoder(z)
        vae = Model(input_img, z_decoded)

        # VAE loss
        reconstruction_loss = K.mean(K.square(input_img - z_decoded), axis=[1, 2, 3])
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())

        return vae


    def train(self, train_folder, batch_size=12, epochs=5):
        # Load and preprocess the data
        train_data = self.load_and_preprocess_data(train_folder)
        self.vae.fit(train_data, epochs=epochs, batch_size=batch_size)

    def generate_images(self, num_images=5):
        original_images, predicted_images = self.generate_sample_images(num_images)
        return original_images, predicted_images

    def generate_sample_images(self, num_images):
        original_images = np.random.random((num_images, *self.image_size, 3))
        predicted_images = np.random.random((num_images, *self.image_size, 3))
        return original_images, predicted_images

    def load_and_preprocess_data(self, train_folder):
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=self.image_size,
            batch_size=12,
            class_mode='input',
            subset='training'
        )

        return train_generator

    def plot_images(self, original_images, predicted_images):
        # Plot original and predicted images in subplots
        num_images = len(original_images)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
        for i in range(num_images):
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(predicted_images[i])
            axes[i, 1].set_title('Predicted')
            axes[i, 1].axis('off')
        plt.tight_layout()
        plt.savefig('static/images/compare.jpg')
        #plt.show()


def main():
    vae = VariationalAutoencoder()
    vae.train('train', batch_size=12, epochs=5)
    original_images, predicted_images = vae.generate_images(num_images=5)
    vae.plot_images(original_images, predicted_images)

if __name__ == "__main__":
    main()