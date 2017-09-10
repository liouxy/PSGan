from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
import time
import collections
import os


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to input tfrecords")
parser.add_argument("--mode", required=True, choices=["train","test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoints")

parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images ever display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps")

parser.add_argument("--batch_size",type=int, default=1, help="number of images in batch")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--ndf", type=int, default=32, help="number of generator filters in first conv layer")
a=parser.parse_args()

EPS = 1e-12

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
def conv(batch_input, kernel_size, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        #padded_input = tf.pad(batch_input,[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
        #conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        conv = tf.nn.conv2d(batch_input, filter, [1,stride, stride, 1], padding='SAME')
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x=tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset",[channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0,0.02))
        mean, variance = tf.nn.moments(input,axes=[0,1,2], keep_dims=False)
        variance_epsilon= 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def strided_conv(batch_input, kernel_size, out_channels):
    with tf.variable_scope("strided_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        strided_conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height*2, in_width*2, out_channels], [1,2,2,1], padding='SAME')
        return strided_conv

def create_generator(generator_inputs1, generator_inputs2, generator_outputs_channels):
    layers=[]
    with tf.variable_scope("encoder_1_1"):
        output = conv(generator_inputs1, 3, 32, 1)
        layers.append(output)
    with tf.variable_scope("encoder_2_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 32, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_3_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)
        layers.append(convolved)

    with tf.variable_scope("encoder_1_2"):
        output = conv(generator_inputs2, 3, 32, 1)
        layers.append(output)
    with tf.variable_scope("encoder_2_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 32, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_3_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)
        layers.append(convolved)

    concat1 = tf.concat([layers[-1], layers[-1-3]], 3)
    with tf.variable_scope("encoder_4"):
        rectified = lrelu(concat1, 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_5"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_6"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 2)
        layers.append(convolved)

    with tf.variable_scope("decoder_7"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 1, 256, 1)
        layers.append(convolved)

    with tf.variable_scope("decoder_8"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 1)
        layers.append(convolved)

    with tf.variable_scope("decoder_9"):
        rectified = lrelu(layers[-1], 0.2)
        strided_convolved = strided_conv(rectified, 2, 128)
        layers.append(strided_convolved)

    concat2 = tf.concat([layers[-1], layers[-1-4]], 3)

    with tf.variable_scope("decoder_10"):
        rectified = lrelu(concat2, 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)

    with tf.variable_scope("decoder_11"):
        rectified = lrelu(layers[-1], 0.2)
        strided_convolved = strided_conv(rectified, 2, 128)
        layers.append(strided_convolved)

    concat3 = tf.concat([layers[-1], layers[-1-8], layers[-1-11]], 3)

    with tf.variable_scope("decoder_12"):
        rectified = lrelu(concat3, 0.2)
        convolved = conv(rectified, 3, 64, 1)
        layers.append(convolved)

    with tf.variable_scope("decoder_13"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, generator_outputs_channels, 1)
        layers.append(convolved)

    return layers[-1]

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], 3)

        with tf.variable_scope("layer_1"):
            convolved = conv(input, 3, a.ndf, 2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], 3, out_channels, stride=stride)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, 3, 1, 1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs,out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_generator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_generator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake+EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real = predict_real,
        predict_fake = predict_fake,
        discrim_loss = ema.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        gen_loss_GAN = ema.average(gen_loss_GAN),
        gen_loss_L1 = ema.average(gen_loss_L1),
        gen_grads_and_vars = gen_grads_and_vars,
        outputs= outputs,
        train = tf.group(update_losses, incr_global_step, gen_train),
    )


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)


