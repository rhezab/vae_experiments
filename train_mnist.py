from vae import VAE
from conv_v1 import conv_v1
from conv_v2 import conv_v2

which_dataset = "fashion"
which_model = "conv_v2"

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

if which_model is "conv_v1":
    model_architecture = conv_v1(latent_dim=10)
elif which_model is "conv_v2":
    model_architecture = conv_v2(latent_dim=10, num_filters=64,
                                 dense_dim=64, conv_layers=2,
                                 pool='max')
vae = VAE(model=model_architecture)
vae.train(x_train, x_test, epochs=10, batch_size=100)
vae.save('./models/conv_v2_mnist_t1_10e')
if latent_dim is 2:
    vae.sample_latent()
vae.sample_test(x_test)
