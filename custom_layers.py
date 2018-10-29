from keras import backend as K
import keras

class xentKLLossLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded, z_mean, z_log_var):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x, z_decoded, z_mean, z_log_var = inputs
        loss = self.vae_loss(x, z_decoded, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x

class SamplingLayer(keras.layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        super(SamplingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    def get_config(self):
        config = {'latent_dim': self.latent_dim}
        base_config = super(SamplingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

