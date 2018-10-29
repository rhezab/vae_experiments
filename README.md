I first became interested in VAEs after reading Nielsen's *[Using Artificial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/).* In particular, I was excited by the idea of interpolating in the latent space of VAEs (and how this is kind of like interpolating in the space of existing human thought). 

So I decided to learn about VAEs and do some experiments on my own, starting with the `mnist` and `fashionmnist` datasets because they seem the most manageable. My implementation follows (but is different from) the one in Chollet's *Deep Learning with Python,* and my code to sample the latent space borrows heavily from his [blog post](https://blog.keras.io/building-autoencoders-in-keras.html) (so it's not very original). 

Below are some interesting results in traversing the latent space of `fashionmnist`. The bottom row shows the original images, and the top row shows the VAE's reconstructions — traversing the latent space from the image on the left to the image on the right (directly above each image is its direct reconstruction).

<div class="imgcapleft">
<img src="/results/inter10.png" style="border:none; width:10%;">
</div>

<div class="imgcapleft">
<img src="/results/inter3.png" style="border:none; width:10%;">
</div>

<div class="imgcapleft">
<img src="/results/inter2.png" style="border:none; width:10%;">
</div>



## Project Structure
- `vae.py` contains the VAE class, which contains all the methods you could do with VAEs (e.g. train, sample latent space, save, etc.)
- `conv_v1.py` and `conv_v2.py` contain the model architectures, which you can use to initalise a VAE class instance and train
- `train_mnist.py` contains a training script
- `sample.py` contains a script for sampling from the latent space (interpolating between two randomly chosen examples)
- `./results` contains some results from interpolating between fashionmnist pictures in the latent space

## Goals
1. Get decent reconstructions
2. Get a decent latent space (with as few dimensions as possible)
3. Build an interface for traversing latent space
4. Add the ability to do "latent space arithmetic", a la Nielsen in [AIA](https://distill.pub/2017/aia/)
5. Do steps 1-4 for more interesting, ambitous data
6. Try to interpret my models

## Data (in order of ambitousness)
- mnist
- fashionmnist
- [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)

## Resources
I learned about VAEs from the Chollet book Python From Deep Learning; and my code for sampling the latent space borrows heavily from Chollet's blog post.

- [Autoencoders with Keras by ramhiser](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/)
- [Building autoencoders in Keras by Chollet](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Autoencoders from Deep Learning Book by Goodfellow, Bengio, Courville](http://www.deeplearningbook.org/contents/autoencoders.html)
- [KL Divergence Explained by countbayesie](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

## Log / Working Notes
- 'conv_v1_mnist_t2'
    - latent_dim: 2
    - num_filters: 64 (constant for all conv layers)
    - dense_dim: 64 (same for decoder and encoder)
    - kernel_size: (3,3)
    - strides: (2,2) (should probably try (1,1) strides with pooling layers or (2,2) strides in between)
    - conv_layers: 2 (same for decoder and encoder)
    - input_shape: (28, 28, 1)
    - after 50 epochs, loss and val loss approx 0.17
- 'conv_v1_fashionmnist_t3'
    - latent_dim: 10
    - else, same as above
    - after 10 epochs, loss and val loss approx 0.28 (already this after 5 epochs)
    - after 30 epochs, loss and val loss approx 0.276
- 'conv_v2_fashionmnist_t1'
    - same as above, but with strides (1,1) in encoder and pool layers in-bewteen
    - ~ 5:40 per epoch for 348,565 trainable params


