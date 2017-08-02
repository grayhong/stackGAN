from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from models import *
from utils import Annotated_MNIST


class Trainer(object):
    def __init__(self, sess, N_g = 28, N_z = 100, N_d = 28, N_embed = 300, dataset_name='mnist'
     W_o = 28, H_o = 28, c_dim = 1, gfc_dim = 128, ld = 1, batch_size = 64):
        self.sess = sess
        self.N_g = N_g
        self.N_z = N_z
        self.N_d = N_d
        self.N_embed = N_embed
        self.W_o = W_o
        self.H_o = H_o
        self.c_dim = c_dim


        self.gfc_dim = gfc_dim
        self.batch_size = batch_size

        self.dataset_name = dataset_name

        self.ld = ld

        self.N_s1_img = self.N_d * self.N_d * self.c_dim

        if self.dataset_name == 'mnist':
            self.annotated_MNIST = Annotated_MNIST(train=True);
            self.total_sample = self.annotated_MNIST.get_sample_num()


        self.build_model()

    def get_noise(self, batch_size):
        multNormal = ds.MultivariateNormalDiag(tf.zeros([self.N_z]), tf.ones([self.N_z]))
        s1gen_noise = multNormal.sample_n(self.batch_size) # batch x N_z
        print('s1gen_noise : {}'.format(self.s1gen_noise))
        return s1gen_noise

    def train(self, total_epoch=100, sample_size = 10):
        self.train_D = tf.train.AdamOptimizer(1e-4, beta1=0.5)\
                                .minimize(-self.loss_D, var_list = self.d_vars)
        self.train_G = tf.train.AdamOptimizer(2e-4, beta1=0.5)\
                                .minimize(-self.loss_G, var_list = self.g_vars)

        total_batch = int(self.total_sample/self.batch_size)
        loss_val_D, loss_val_G = 0, 0


        self.sess.run(tf.global_variables_initializer())
        for epoch in range(total_epoch):
            for i in range(total_batch):
                sent, batch_x, batch_y = self.annotated_MNIST(self.batch_size)
                noise = self.get_noise(self.batch_size)

                _, loss_val_D = self.sess.run([self.train_D, self.loss_D], feed_dict = {self.sent: sent, self.x: batch_x, self.z: noise})
                _, loss_val_G = self.sess.run([self.train_G, self.loss_G], feed_dict = {self.sent: sent, self.z: noise})

            print('Epoch: ', '%04d' % epoch,
                  'D loss: {:.4}'.format(loss_val_D),
                  'G loss: {:.4}'.format(loss_val_G))

            #if epoch == 0 or (epoch + 1) % 10 == 0:
            noise = self.get_noise(self.sample_size)
            sent_indices = self.annotated_MNIST.generate_sentences(sample_size)
            samples = self.sess.run(self.s1_G_z, feed_dict={self.z: noise, self.sent: sent_indices})
            sentences = self.annotated_MNIST.conver_to_idx(sent_indices)


            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)), cmap='Greys')
                ax[i].title(sentences[i])

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)


    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.N_z])
        self.sent = tf.placeholder(tf.float32, [None, self.N_embed]) # idx seq x batch_size
        self.x = tf.placeholder(tf.float32, [None, self.N_s1_img])

        init_width = 0.5 / self.embed_size
        self.embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -init_width, init_width), name='embed')

        # embedding
        self.sent_embed_seq = tf.nn.embedding_lookup(self.embed, self.sent, name='sent_embed')
        self.sent_embed = tf.reduce_sum(self.sent_embed_seq, 1, keep_dims = True)

        # conditional augmentation
        self.multNormal1 = ds.MultivariateNormalDiag(tf.zeros([self.N_z]), tf.ones([self.N_z]))
        self.s1_c, self.s1_mu_o, self.s1_sigma_o = condAugment(self.sent_embed, self.multNormal1, 's1')
        self.multNormal2 = ds.MultivariateNormalDiag(self.s1_mu_o, self.s1_sigma_o)
        self.kl = tf.contrib.distributions.kl_divergence(self.multNormal2, self.multNormal1);


        # concat with noise
        self.s1_G_input = tf.concat([self.z, self.s1_c], 1) # batch x (100 + 128)
        self.s1_G_z, self.s1_G_vars = stage1_generator(self.s1_G_input, self.z)
        self.s1_D_G_z, self.s1_D_vars = stage1_discriminator(self.sent_embed, self.G_z)
        self.s1_D_x, self.s1_D_vars = stage1_discriminator(self.sent_embed, self.x, reuse = True)

        # loss
        self.loss_D = tf.reduce_mean(tf.log(self.s1_D_x) + tf.log(1 - self.s1_D_G_z))
        self.loss_G = tf.reduce_mean(tf.log(1 - self.s1_D_G_z)) + self.ld * self.kl;

        t_var = tf.trainable_variables()

        self.s1g_vars = [v for v in t_var if 's1g_' in v.name]
        self.s1d_vars = [v for v in t_var if 's1d_' in v.name]
        print(self.s1g_vars)
        print(self.s1d_vars)

        self.saver = tf.train.Saver(max_to_keep = 1)
