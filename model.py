#!/usr/bin/env python3
"""Variational Autoencoder models.

Available latent distributions:
    * Gaussian/ Normal [1]
    * RelaxedBernoulli/ BinConcrete [2]

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
[2] Maddison, Chris J., Andriy Mnih, and Yee Whye Teh. "The concrete
    distribution: A continuous relaxation of discrete random variables."

Author, Karen Ullrich June 2019
"""

import tensorflow as tf
import tensorflow_probability as tfp

from utils import bm_transform, binary2continuous

tfd = tfp.distributions
NUM_MBITS = 23 # mantissa bits, for float 32 this is 23 bits

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, out_dim_infernce_net=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        if out_dim_infernce_net is None:
            out_dim_infernce_net = 2 * latent_dim

        # This is heavily inspired by an official VAE tutorial,
        # https://www.tensorflow.org/beta/tutorials/generative/cvae
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2),
                    activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2),
                    activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(400, activation='relu'),
                # No activation
                tf.keras.layers.Dense(out_dim_infernce_net)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(400, activation='relu'),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def encode(self, x):
        pass

    def decode(self, x, z, return_probs=False):
        logits = self.generative_net(z)

        # Bernoullli observation model is equivalent to cross entropy loss
        observation_dist = tfd.Bernoulli(logits=logits)
        if return_probs:
            return observation_dist.probs_parameter()
        logpx_z = tf.reduce_sum(observation_dist.log_prob(x), axis=[1, 2, 3])
        return logpx_z


class NVAE(VAE):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.prior = tfd.Normal(loc=tf.zeros(latent_dim),
                                scale=tf.ones(latent_dim))
        self.prior_sample_fun = self.prior.sample

    def encode(self, x):
        loc, logvar = tf.split(self.inference_net(x), num_or_size_splits=2,
                               axis=1)
        latent_dist = tfd.Normal(loc=loc, scale=tf.exp(logvar))
        latent_samples = latent_dist.sample()
        return latent_samples, latent_dist.log_prob(
            latent_samples), self.prior.log_prob(latent_samples)

class BVAE(VAE):
    def __init__(self, latent_dim, prior_temperature=0.1):
        super().__init__(latent_dim, latent_dim)
        
        # even probability for 0 and 1
        # note that the prior has a temperature, that does not need to be
        # the same as the posterior
        probs = 0.5 * tf.ones(latent_dim)
        self.prior = tfd.Logistic(tf.math.log(probs) / prior_temperature,
                                  1. / prior_temperature)
        self.prior_sample_fun = lambda x: tf.sigmoid(self.prior.sample(x))

    def encode(self, x, temperature=0.2):
        logits = self.inference_net(x)
        # The temperature adjusts the relaxiation of the Concrete 
        # distribution. We use,
        latent_dist = tfd.Logistic(logits / temperature, 1. / temperature)
        # instead of
        # tfd.RelaxedBernoulli(temperature=temperature, logits=logits).
        # Otherwise we run into underflow issues when computing the 
        # log_prob. This has been expained in [2] Appendix C.3.2.

        logistic_samples = latent_dist.sample()
        log_qz = latent_dist.log_prob(logistic_samples)
        log_pz = self.prior.log_prob(logistic_samples)
        return tf.sigmoid(logistic_samples), log_qz, log_pz

class BNVAE(VAE):
    # VAE with visible binary sampling
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.prior = tfd.Normal(loc=tf.zeros(latent_dim),
                                scale=tf.ones(latent_dim))
        self.prior_sample_fun = self.prior.sample

    def reparameterize(self, loc, scale):

        # classic re-parametrization trick but we start with Bernoulli samples

        # Bernoulli smaples with shape [batch_size, latent_dim, NUM_MBITS]
        probs = 0.5 * tf.ones(loc.shape + (NUM_MBITS,))
        latent_dist = tfd.Bernoulli(probs=probs, dtype=loc.dtype)
        bernoulli_noise = latent_dist.sample()

        # turn Bernoulli noise into uniform noise
        # shape [batch_size, latent_dim]
        uniform_noise = binary2continuous(bernoulli_noise)
        # turn uniform noise into Gaussian noise with Box-Muller transform
        # shape [batch_size, latent_dim]
        gaussian_noise = bm_transform(uniform_noise)
        # re-parametrization trick
        return loc + gaussian_noise * scale


    def encode(self, x):
        loc, logvar = tf.split(self.inference_net(x), num_or_size_splits=2,
                               axis=1)
        latent_dist = tfd.Normal(loc=loc, scale=tf.exp(logvar))
        # replace the normal sampling
        # -> latent_samples = latent_dist.sample()
        # with equivalent sampling (does exactly the same as before!)
        latent_samples = self.reparameterize(loc=loc, scale=tf.exp(logvar))
        log_qz = latent_dist.log_prob(latent_samples)
        log_pz = self.prior.log_prob(latent_samples)
        return latent_samples, log_qz, log_pz


class BCVAE(VAE):
    # binary to continious VAE
    def __init__(self, latent_dim, prior_temperature=1.0, amortize_linear_transform = False):


        self.amortize_linear_transform = amortize_linear_transform

        if self.amortize_linear_transform:
            out_dim_infernce_net = (NUM_MBITS + 1 + 1) * latent_dim
        else:
            out_dim_infernce_net = NUM_MBITS * latent_dim

        super().__init__(latent_dim = latent_dim,
                         out_dim_infernce_net=out_dim_infernce_net)


        probs = 0.5 * tf.ones((latent_dim, NUM_MBITS))
        self.prior = tfd.Logistic(tf.math.log(probs) / prior_temperature,
                                  1. / prior_temperature)
        self.prior_sample_fun = lambda x: self.continuify(
            tf.sigmoid(self.prior.sample(x)))

    def continuify(self, binconcrete_samples, loc=None, scale=None):

        learned_noise = binary2continuous(binconcrete_samples) # [0,1)
        # send through Box-Muller transform (might not be nessesary)
        learned_noise = bm_transform(learned_noise)

        out = learned_noise
        if scale is not None: out = out * scale
        if loc is not None: out = out + loc
        return out


    def encode(self, x, temperature=1.0):

        if  self.amortize_linear_transform:
            # split output of inference network
            splits = [ NUM_MBITS * self.latent_dim, self.latent_dim, self.latent_dim]
            logits, loc, logvar = tf.split(self.inference_net(x), splits, axis=1)
            scale = tf.exp(logvar)
        else:
            logits = self.inference_net(x)
            loc, scale = None, None
        logits = tf.reshape(logits,(-1, self.latent_dim, NUM_MBITS))

        # generate BinConcrete samples
        latent_dist = tfd.Logistic(logits / temperature, 1. / temperature)
        logistic_samples = latent_dist.sample()
        binconcrete_samples = tf.sigmoid(logistic_samples)

        # BinConcrete (approx. Bernoulli) samples into samples from learned
        # continuous distribution
        samples = self.continuify(binconcrete_samples, loc, scale)

        # continue further as in BVAE
        flat_shape = (-1,self.latent_dim * NUM_MBITS)
        logqz_x = tf.reshape(latent_dist.log_prob(logistic_samples), flat_shape)
        logpz = tf.reshape(self.prior.log_prob(logistic_samples), flat_shape)
        return samples, logqz_x, logpz




