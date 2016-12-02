# GANs

This repository implements feedforward [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf) and an experimental form of recurrent GAN. It includes some of [the work done at OpenAI](https://arxiv.org/pdf/1606.03498v1.pdf) on GANs.

# Basic results

As a quick test, I trained a small GAN to produce MNIST digits. Here is the result:

![MNIST renderings](demo/mnist_gen/renderings.png)

I also built a deeper GAN for extracting features from face images. The results can be found in [MustacheMash](https://github.com/unixpickle/mustachemash)

# Recurrent GANs?

After a few different attempts, I still have not managed to get a recurrent GAN to produce reasonably English-sounding text. I will probably pick this up again in the future.
