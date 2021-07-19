import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.keras.initializers import RandomNormal

# generator
def build_gen_loss(disc):
    @tf.function
    def gen_loss(y_true, gen_imgs):
        disc_preds = disc(gen_imgs)
        return binary_crossentropy(np.ones(disc_preds.shape), disc_preds)
    return gen_loss
        
def build_generator(latent_dim, disc):
    model = Sequential()
    # foundation for 8x8 image
    n_nodes = 256 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x48
    model.add(Conv2DTranspose(128, (4,4), strides=(2,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 128x96
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=build_gen_loss(disc), optimizer=opt)
#     model.compile(loss=binary_crossentropy, optimizer=opt)

    return model


# discriminator
def build_discriminator(in_shape=(128,96,3)):
    init = RandomNormal(stddev=0.02)
    
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def build_composite(latent_dim):
    disc = build_discriminator()
    gen = build_generator(100, disc)
    return gen, disc
