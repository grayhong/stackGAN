
import tensorflow as tf

from ops import *

ds = tf.contrib.distributions

def deconv_size(size, stride):
    # padding outside by 1, with dilation size 1 , 3x3 -> 7x7 then filter with 3x3 -> 5x5
    # apply (W - F + 2P) / S = output_size
    # 64 -> 32 with stride 2
    return int(math.ceil(float(size) / float(stride)))

class StackGAN(object):
    def __init__(self, sess, N_g = 128, N_z = 100, N_d = 128, N_embed = 300, W_o = 64, H_o = 64, c_dim = 3, gfc_dim = 1024, ld = 1, batch_size = 64):
        self.N_g = N_g
        self.N_z = N_z
        self.N_d = N_d
        self.N_embed = N_embed
        self.W_o = W_o
        self.H_o = H_o
        self.c_dim = c_dim
        self.gfc_dim = gfc_dim
        self.batch_size = batch_size

        self.ld = ld

        self.N_s1_img = self.N_d * self.N_d * self.c_dim

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.N_z])
        self.sent = tf.placeholder(tf.float32, [None, self.N_embed])
        self.x = tf.placeholder(tf.float32, [None, self.N_s1_img])

        self.cond_aug = self.condAugment(self.sent, 's1')
        self.G_z = self.stage1_generator(self.cond_aug, self.z)
        self.D_G_z = self.stage1_discriminator(self.sent, self.G_z)
        self.D_x = self.stage1_discriminator(self.sent, self.x, reuse = True)

        ## TODO : finish loss ftn (add KL divergence)
        self.loss_D = tf.reduce_mean(tf.log(self.D_x) + tf.log(1 - self.D_G_z))
        self.loss_G = tf.reduce_mean(tf.log(1 - self.D_G_z)) + self.ld *

        t_var = tf.trainable_variables()

        self.s1g_vars = [v for v in t_var if 's1g_' in v.name]
        self.s1d_vars = [v for v in t_var if 's1d_' in v.name]
        print(self.s1g_vars)
        print(self.s1d_vars)

        self.saver = tf.train.Saver(max_to_keep = 1)

    def training():



    def condAugment(self, embed, scope):

        with tf.variable_scope(scope or 'condAugment'):
            ## make mu, sigma
            self.cond_gen = tf.reshape(linear(embed, self.N_g * 2, scope='condAugmentGen'), [-1, self.N_g, 2]) # batch x N_g x 2
            self.cond_mu_o = self.cond_gen[0] # batch x N_g
            self.cond_sigma_o = self.cond_gen[1] # batch x N_g
            ## do reparametrization
            self.multNormal1 = ds.MultivariateNormalDiag(self.cond_mu_0, self.cond_sigma_o)
            self.cond_epsilon = self.multNormal1.sample_n(self.batch_size) # batch x N_g
            print('cond_epsilon : {}'.format(self.cond_epsilon))
            self.cond_c_o = self.cond_mu_o + tf.multiply(self.cond_sigma_o, self.cond_epsilon)

            return self.cond_c_o


    # TODO: finish generator with nearest neighbor upscaling
    def stage1_generator(self, c):
        #
        # g_h1, g_w1 = deconv_size(self.H_o, 2), deconv_size(self.W_o, 2) # 64 x 64 -> 32 x 32
        # g_h2, g_w2 = deconv_size(g_h1, 2), deconv_size(g_w1, 2) # 32 x 32 -> 16 x 16

        with tf.variable_scope('s1_generator') as scope:
            # maybe we can try tf.image.resize_nearest_neighbor as in the paper

            ## concat with noise
            multNormal2 = ds.MultivariateNormalDiag(tf.zeros([self.N_z]), tf.ones([self.N_z]))
            self.s1gen_noise = multNormal2.sample_n(self.batch_size) # batch x N_z
            print('s1gen_noise : {}'.format(self.s1gen_noise))
            self.s1gen_input = tf.concat([self.s1gen_noise, self.s1gen_input], 1) # batch x (100 + 128)


    # it actually doesn't matter to use dcgan generator or stackGAN generator
    def stage1_DCGANgenerator(self, c, z):
        g_h4, g_w4 = deconv_size(self.H_o, 2), deconv_size(self.W_o, 2)
        g_h3, g_w3 = deconv_size(g_h4, 2), deconv_size(g_w4, 2)
        g_h2, g_w2 = deconv_size(g_h3, 2), deconv_size(g_w3, 2)
        g_h1, g_w1 = deconv_size(g_h2, 2), deconv_size(g_w2, 2)

        with tf.variable_scope('s1_generator') as scope:
            ## concat with noise
            multNormal2 = ds.MultivariateNormalDiag(tf.zeros([self.N_z]), tf.ones([self.N_z]))
            self.s1g_noise = multNormal2.sample_n(self.batch_size) # batch x N_z
            print('s1gen_noise : {}'.format(self.s1g_noise))
            self.s1g_input = tf.concat([self.s1g_noise, c], 1) # batch x (100 + 128)

            self.s1g_lin_noise = tf.nn.relu(linear(self.s1g_input, g_h1 * g_w1 * self.gfc_dim, 's1g_lin_1'))
            self.s1g_reshape = tf.reshape(self.s1g_lin_noise, [-1, g_h1, g_w1, self.gfc_dim])
            self.s1g_batch_relu = tf.nn.relu(self.s1g_batch_norm_0(self.s1g_reshape))

            self.s1g_deconv_1 = deconv2d(self.s1g_batch_relu, [self.batch_size, g_h2, g_w2, int(self.gfc_dim / 2)], name = 's1g_deconv_1')
            self.s1g_deconv_1_batch_relu = tf.nn.relu(self.s1g_batch_norm_1(self.s1g_deconv_1))

            self.s1g_deconv_2 = deconv2d(self.s1g_deconv_1_batch_relu, [self.batch_size, g_h3, g_w3, int(self.gfc_dim / 4)], name = 's1g_deconv_2')
            self.s1g_deconv_2_batch_relu = tf.nn.relu(self.s1g_batch_norm_2(self.s1g_deconv_2))

            self.s1g_deconv_3 = deconv2d(self.s1g_deconv_2_batch_relu, [self.batch_size, g_h4, g_w4, int(self.gfc_dim / 8)], name = 's1g_deconv_3')
            self.s1g_deconv_3_batch_relu = tf.nn.relu(self.s1g_batch_norm_3(self.s1g_deconv_3))

            self.s1g_deconv_4 = deconv2d(self.s1g_deconv_3_batch_relu, [self.batch_size, self.H_o, self.W_o, self.c_dim], name = 's1g_deconv_4')
            self.s1g_deconv_4_tanh = tf.nn.tanh(self.s1g_deconv_4)
            print('shape of generated tensor : {}'.format(self.s1g_deconv_4_tanh))

            return self.s1g_deconv_4_tanh

    def stage1_discriminator(self, embed, image, reuse = False):
        with tf.variable_scope('s1_discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            # reshape the image if needed
            image = tf.reshape(image, [-1, self.W_o, self.H_o, self.c_dim])

            self.s1d_conv_1 = conv2d(image, int(self.df_dim / 8), name = 's1d_conv_1')
            self.s1d_conv_1_batch_lrelu = leakyReLU(self.s1d_conv_1)

            self.s1d_conv_2 = conv2d(self.s1d_conv_1_batch_lrelu, int(self.df_dim / 4), name = 's1d_conv_2')
            self.s1d_conv_2_batch_lrelu = leakyReLU(self.s1d_batch_norm_1(self.s1d_conv_2))

            self.s1d_conv_3 = conv2d(self.s1d_conv_2_batch_lrelu, int(self.df_dim / 2), name = 's1d_conv_3')
            self.s1d_conv_3_batch_lrelu = leakyReLU(self.s1d_batch_norm_2(self.s1d_conv_3))

            self.s1d_conv_4 = conv2d(self.s1d_conv_3_batch_lrelu, self.df_dim, name = 's1d_conv_4')
            self.s1d_conv_4_batch_lrelu = leakyReLU(self.s1d_batch_norm_3(self.s1d_conv_4))
            print('after conv : {}'.format(self.s1d_conv_4_batch_lrelu))

            self.s1d_rep = linear(embed, self.N_d, 's1d_lin_1')
            self.s1d_tile = tf.reshape(tf.tile(self.s1d_rep, [1, self.M_d, self.M_d, 1]), [self.M_d, self.M_d])
            print('tiled one : {}'.format(self.s1d_tile))

            self.s1d_img_text = tf.concat(3, [self.s1d_conv_4_batch_lrelu, self.sld_tile])

            self.s1d_1conv = conv2d(self.s1d_img_text, self.df_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='s1d_1conv')
            self.s1d_1conv_batch_lrelu = leakyReLU(self.s1d_batch_norm_4(self.s1d_1conv))

            self.s1d_1conv_lin = linear(self.s1d_1conv, 1, 's1d_1conv_lin')

            return tf.nn.sigmoid(self.s1d_1conv_lin)
