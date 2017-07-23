import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


class batch_norm(object):
    def __init__(self, epsilon = 1e-5, momentum = 0.5, name = "batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, train = True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
       w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
       initializer=tf.truncated_normal_initializer(stddev=stddev))

       conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
       biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
       conv = tf.nn.bias_add(conv, biases)

       return conv

def deconv2d(input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer = tf.random_normal_initializer(stddev = stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides = [1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        return deconv

def leakyReLU(x, leak = 0.2, name="leakyReLU"):
    return tf.maximum(x, leak * x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
