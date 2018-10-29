from vae import VAE

which_dataset = 'fashion'
model_fpath = './models/conv_v1_fashionmnist_t3_30e.h5'

if which_dataset is "digits":
    from keras.datasets import mnist
    (x_train, _), (x_test, y_test) = mnist.load_data()
elif which_dataset is "fashion":
    from keras.datasets import fashion_mnist
    (x_train, _), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae = VAE(fpath=model_fpath)
# vae.sample_latent()
# vae.sample_test(x_test)
localvs = vae.interpolate(x_test)
