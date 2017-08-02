import os
import numpy as np
import tensorflow as tf

from trainer import Trainer

def main():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    FLAGS = flags.FLAGS
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        stackGAN = Trainer(sess)
        stackGAN.train()


if __name__ == '__main__':
    main()
