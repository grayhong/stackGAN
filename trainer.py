from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image
import Annotated_MNIST


class Trainer(object):
    def __init__(self, sess, N_g = 28, N_z = 100, N_d = 28, N_embed = 300,
     W_o = 28, H_o = 28, c_dim = 1, gfc_dim = 128, ld = 1, batch_size = 64):
        self.N_g = N_g
        self.N_z = N_z
        self.N_d = N_d
        self.N_embed = N_embed
        self.W_o = W_o
        self.H_o = H_o
        self.c_dim = c_dim

        self.s = W_o
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)


        self.gfc_dim = gfc_dim
        self.batch_size = batch_size

        self.ld = ld

        self.N_s1_img = self.N_d * self.N_d * self.c_dim

        if self.dataset_name == 'mnist':
            self.input_sample = input_data.read_data_sets("./mnist/data/", one_hot=True)
            self.total_sample = self.input_sample.train.num_examples
            self.annotated_MNIST = Annotated_MNIST(train=False);


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


    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = GeneratorCNN(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x
