import keras
# from keras import backend as K
# from keras import layers
from keras import Model

import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from custom_layers import xentKLLossLayer, SamplingLayer
from custom_fns import none_loss
from conv_v1 import conv_v1

class VAE(object):
    def __init__(self, fpath=None, model=None):
        if fpath:
            custom_objects={
                'xentKLLossLayer': xentKLLossLayer,
                'SamplingLayer': SamplingLayer,
                'none_loss': none_loss
            }
            self.model = keras.models.load_model(fpath, custom_objects=custom_objects)
        elif model:
            self.model = model
        self.encoder = Model(self.model.input,
                             self.model.get_layer('latent_sampling').output)
        self.decoder = self.model.get_layer('decoder')

    def train(self, x_train, x_test, epochs=10, batch_size=16, plot=True):
        history = self.model.fit(x=x_train, 
                                 y=x_train,
                                 # y=None,
                                shuffle=True,
                                epochs=epochs,
                                batch_size=batch_size,
                                # validation_data=(x_test, None),
                                validation_data=(x_test, x_test))
        metrics = ['loss', 'val_loss']
        log_= {}
        for metric in metrics:
            log_[metric] = history.history[metric]
        self.log = pd.DataFrame(log_)
        if plot is True:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def encode_decode(self, input_vectors):
        encodings = self.encoder.predict(input_vectors)
        return self.decoder.predict(encodings)

    def sample_latent(self, n=10, figsize=(10,10), img_size=28):
        figure = np.zeros((img_size * n, img_size * n))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([xi,yi])
                # z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
                x_decoded = self.decoder.predict(z_sample[None])
                # x_decoded = self.encode_decode(z_sample[None])
                img = x_decoded[0].reshape(img_size, img_size)
                figure[i * img_size: (i+1) * img_size,
                       j * img_size: (j+1) * img_size] = img
        plt.figure(figsize=figsize)
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    def sample_test(self, test_x, figsize=(10,10), num_samples=2, img_size=28):
        num_classes = 10
        random_idxs = [np.random.randint(0, 1000) for i in range(num_samples)]
        xs = []
        for i in range(num_classes):
            for j in random_idxs:
                xs.append(test_x[1000*i + j])
        # decoded_xs = self.model.predict(np.array(xs))
        decoded_xs = self.encode_decode(np.array(xs))
        n = num_classes * num_samples
        plt.figure(figsize=figsize)
        for i in range(n):
            ax = plt.subplot(2, n, i+1)
            plt.imshow(xs[i].reshape(img_size,img_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i+1+n)
            plt.imshow(decoded_xs[i].reshape(img_size, img_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def interpolate(self, test_x, figsize=(10,10), num_interpolate=3, img_size=28):
        """
        class_idx = {
            'tshirt': 0,
            'trouser': 1,
            'pullover': 2,
            'dress': 3,
            'coat': 4,
            'sandal': 5,
            'shirt': 6,
            'sneaker': 7,
            'bag': 8,
            'boot': 9
        }
        """
        x1 = np.random.randint(0, 10000)
        x2 = np.random.randint(0, 10000)
        x1 = test_x[x1]
        x2 = test_x[x2]
        diff_vec = x2 - x1
        xs = [x1]
        for i in range(num_interpolate):
            xs.append(x1 + ((1.0 * i/num_interpolate) * diff_vec))
        xs.append(x2)
        decoded_xs = self.encode_decode(np.array(xs))

        n = len(xs)
        plt.figure(figsize=figsize)
        for i in [1, n]:
            ax = plt.subplot(2, n, i+n)
            plt.imshow(xs[i-1].reshape(img_size,img_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        for i in range(n):
            ax = plt.subplot(2, n, i+1)
            plt.imshow(decoded_xs[i].reshape(img_size,img_size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        return locals()

    def save(self, fpath):
        self.log.to_pickle(fpath+'_log.pkl')
        self.model.save(fpath+'.h5')

