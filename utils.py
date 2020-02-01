# TODO make tests!!!!!

import tensorflow as tf
import math as m
TWO_PI = tf.constant(2. * m.pi)
EPSILON = 1e-15


def exponent_bits(num_bits, dtype):
    """ Returns exponents for binary to integer transform.

    e.g. [7,6,5,4,3,2,1,0] for 8 bit integers
    """
    exponent_bits = -tf.range(-(num_bits - 1), 1, dtype=dtype)
    return tf.expand_dims(tf.expand_dims(exponent_bits, 0), 0)



def integer2binary(integer, num_bits=23):
    """ Turn integer tensor to binary representation.

    num_bits: Gives the bit resolution for the integer, int23 as default.
    """

    assert len(integer.shape)==2, "Function can only convert 2-dim tensors atm."

    e_bits = exponent_bits(num_bits, integer.dtype)
    out = tf.expand_dims(integer, -1) / 2 ** e_bits
    return (out - (out % 1)) % 2


def binary2continuous(binary):
    """ Turn binary tensor to continuous representation with interval [0,1).

    If binary is sampled from fair Bernoulli (p=0.5), than output is uniform.
    """

    assert len(binary.shape) == 3, "Function can only convert 3-dim tensors atm."

    num_bits = binary.shape[-1]

    # number of integers that can be generated
    M = 2 ** num_bits + 1 # + 1 guarantees interval [0,1)

    e_bits = exponent_bits(num_bits, binary.dtype)
    return tf.math.reduce_sum(binary * 2 ** e_bits, -1) / M


def bm_transform(uniform_samples):
    """Perform Box-Muller transform.

    Turns samples from a uniform distribution into one from a STD Gaussian one.
    """
    assert uniform_samples.shape[0] % 2 == 0, "Need even number of samples."
    u1, u2 = tf.split(uniform_samples, 2)

    # Box-Muller transform
    R = tf.math.sqrt(- 2. * tf.math.log(u1 + EPSILON) + EPSILON)
    # possibly numerical instable
    assert tf.math.reduce_any(tf.math.is_nan(R)).numpy() == False
    phi = TWO_PI * u2


    z1 = R * tf.math.cos(phi)
    z2 = R * tf.math.sin(phi)
    return tf.concat([z1,z2],0)




def inv_bm_transform(gaussian_samples):
    """Performs the inverse  Box-Muller transform.

    Turns samples from from a STD Gaussian distribution into  a uniform one.
    """
    assert gaussian_samples.shape[0] % 2 == 0, "Need even number of samples."
    z1, z2 = tf.split(gaussian_samples, 2)

    # Box-Muller transform
    R_squ = z1**2 + z2**2
    phi = tf.math.tanh(z2/z1)

    u1 = tf.math.exp(R_squ / -2)
    u2 = phi /TWO_PI
    return tf.concat([u1, u2], 0)