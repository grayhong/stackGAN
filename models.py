import tensorflow as tf

from ops import *

def condAugment(self, embed, normal_dist, scope):

    with tf.variable_scope(scope or 'condAugment'):
        ## make mu, sigma
        cond_gen = tf.reshape(linear(embed, N_g * 2, scope='condAugmentGen'), [-1, N_g, 2]) # batch x N_g x 2
        cond_mu_o = cond_gen[0] # batch x N_g
        cond_sigma_o = cond_gen[1] # batch x N_g
        ## do reparametrization
        cond_epsilon = normal_dist.sample_n(batch_size) # batch x N_g
        print('cond_epsilon : {}'.format(cond_epsilon))
        cond_c_o = cond_mu_o + tf.multiply(cond_sigma_o, cond_epsilon)

        return cond_c_o, cond_mu_o, cond_sigma_o


def stage1_generator(c, z, image_size = [28, 28, 1], gfc_dim = 128):

    s = image_size[0]
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
    c_dim = image_size[2]

    with tf.variable_scope('s1g') as scope:
        # fully connected net with batch norm
        s1g_input = tf.concat([z, c], 1)
        s1g_lin_noise = tf.nn.relu(linear(s1g_input, s16 * s16 * gfc_dim * 8, 's1g_lin_1'))
        s1g_reshape = tf.reshape(s1g_lin_noise, [-1, s16, s16, gfc_dim * 8])
        s1g_batch = s1g_batch_norm_0(s1g_reshape)

        # 1st nearest neighbor upsampling & conv2d & relu & batch
        s1g_nn_1 = tf.image.resize_nearest_neighbor(s1g_batch, [s8, s8], name = 's1g_nn_1')
        s1g_conv_1 = conv2d(s1g_nn_1, gfc_dim * 4, name = 's1g_conv_1')
        s1g_conv_1_batch_relu = tf.nn.relu(s1g_conv_1)

        # 2nd nearest neighbor upsampling & conv2d & relu & batch
        s1g_nn_2 = tf.image.resize_nearest_neighbor(s1g_conv_1_batch_relu, [s4, s4], name = 's1g_nn_2')
        s1g_conv_2 = conv2d(s1g_nn_2, gfc_dim * 2, name = 's1g_conv_2')
        s1g_conv_2_batch_relu = tf.nn.relu(s1g_conv_2)

        # 3rd nearest neighbor upsampling & conv2d & relu & batch
        s1g_nn_3 = tf.image.resize_nearest_neighbor(s1g_conv_2_batch_relu, [s2, s2], name = 's1g_nn_3')
        s1g_conv_3 = conv2d(s1g_nn_3, gfc_dim, name = 's1g_conv_3')
        s1g_conv_3_batch_relu = tf.nn.relu(s1g_conv_3)

        # 4th nearest neighbor upsampling & conv2d & relu & batch
        s1g_nn_4 = tf.image.resize_nearest_neighbor(s1g_conv_3_batch_relu, [s, s], name = 's1g_nn_4')
        s1g_conv_4 = conv2d(s1g_nn_4, c_dim, name = 's1g_conv_4')
        s1g_conv_4_batch_relu = tf.nn.relu(s1g_conv_4)

    variables = tf.contrib.framework.get_variables(scope)
    return s1g_conv_4_batch_relu, variables

def stage1_discriminator(embed, image, image_size = [28, 28, 1], M_d = 4, N_d = 28, df_dim = 1024, reuse = False):

    W_o, H_o, c_dim = image_size[0], image_size[1], image_size[2]


    with tf.variable_scope('s1d') as scope:
        if reuse:
            scope.reuse_variables()

        # reshape the image if needed
        image = tf.reshape(image, [-1, W_o, H_o, c_dim])

        s1d_conv_1 = conv2d(image, int(df_dim / 8), name = 's1d_conv_1')
        s1d_conv_1_batch_lrelu = leakyReLU(s1d_conv_1)

        s1d_conv_2 = conv2d(s1d_conv_1_batch_lrelu, int(df_dim / 4), name = 's1d_conv_2')
        s1d_conv_2_batch_lrelu = leakyReLU(s1d_batch_norm_1(s1d_conv_2))

        s1d_conv_3 = conv2d(s1d_conv_2_batch_lrelu, int(df_dim / 2), name = 's1d_conv_3')
        s1d_conv_3_batch_lrelu = leakyReLU(s1d_batch_norm_2(s1d_conv_3))

        s1d_conv_4 = conv2d(s1d_conv_3_batch_lrelu, df_dim, name = 's1d_conv_4')
        s1d_conv_4_batch_lrelu = leakyReLU(s1d_batch_norm_3(s1d_conv_4))
        print('after conv : {}'.format(s1d_conv_4_batch_lrelu))

        s1d_rep = linear(embed, N_d, 's1d_lin_1')
        s1d_tile = tf.reshape(tf.tile(s1d_rep, [1, M_d, M_d, 1]), [M_d, M_d])
        print('tiled one : {}'.format(s1d_tile))

        s1d_img_text = tf.concat(3, [s1d_conv_4_batch_lrelu, sld_tile])

        s1d_1conv = conv2d(s1d_img_text, df_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='s1d_1conv')
        s1d_1conv_batch_lrelu = leakyReLU(s1d_batch_norm_4(s1d_1conv))

        s1d_1conv_lin = linear(s1d_1conv, 1, 's1d_1conv_lin')

    variables = tf.contrib.framework.get_variables(scope)
    return tf.nn.sigmoid(s1d_1conv_lin), variables
