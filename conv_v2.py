import keras
from keras import backend as K
from keras import layers
from keras import Model

import numpy as np

from custom_layers import xentKLLossLayer, SamplingLayer
from custom_fns import none_loss

# conv w/ (1,1) strides with pooling in-between
def conv_v2(latent_dim=8, num_filters=64, dense_dim=128, kernel_size=(3,3), 
            input_shape=(28,28,1), conv_layers=2, pool='max'):
    encoder_input = keras.Input(shape=input_shape, name='encoder_input')
    x = encoder_input
    for i in range(conv_layers):
        x = layers.Conv2D(num_filters, kernel_size, strides=(1,1),
                          padding='same', activation='relu',
                          name='encoder_conv_'+str(i))(x)
        if pool is 'max':
            x = layers.MaxPooling2D(pool_size=(2,2), name='encoder_maxpool_'+str(i))(x)
        elif pool is 'average':
            x = layers.AveragePooling2D(pool_size=(2,2), name='encoder_avgpool_'+str(i))(x)
        elif pool is 'none':
            x = layers.Conv2D(num_filters, kernel_size, strides=(2,2),
                              padding='same', activation='relu',
                              name='encoder_conv_'+str(i)+'b')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(dense_dim, activation='relu', name='encoder_dense_')(x)
    z_mean = layers.Dense(latent_dim, name='latent_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='latent_var')(x)
    z = SamplingLayer(latent_dim, name='latent_sampling')([z_mean, z_log_var])
    # self.encoder = Model(encoder_input, z, name='encoder')

    decoder_input = layers.Input(K.int_shape(z)[1:], name='decoder_input')
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu', name='decoder_dense')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:], name='reshape')(x)
    for i in range(conv_layers):
        x = layers.Conv2DTranspose(num_filters, kernel_size, strides=(2,2), padding='same',
                                    activation='relu', name='decoder_conv_'+str(i))(x)
    # below squeeze everything to btwn 0-1 grayscale values
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='decoder_conv_to_img')(x) 
    decoder = Model(decoder_input, x, name='decoder')
    z_decoded = decoder(z)
    y = xentKLLossLayer(name='loss')([encoder_input, z_decoded, z_mean, z_log_var])

    model = Model(encoder_input, y, name='vae')
    model.compile(optimizer='rmsprop', loss=none_loss) #loss=None)
    model.summary()
    return model
