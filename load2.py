import tensorflow as tf
import numpy as np
import argparse
import math
import time
import collections
import os
import json
import glob
import random
parser = argparse.ArgumentParser()

parser.add_argument("--train_tfrecord", help="filename of train_tfrecord",default="/data/psgan/train.tfrecords")
parser.add_argument("--test_tfrecord", help="filename of test_tfrecord", default="test.tfrecords")
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

parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
a=parser.parse_args()


Examples = collections.namedtuple("Examples", "imnames, inputs1, inputs2, targets,steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'im_mul_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_raw': tf.FixedLenFeature([], tf.string)
                                       })
    im_mul_raw = tf.decode_raw(features['im_mul_raw'], tf.uint8)
    im_mul_raw = tf.reshape(im_mul_raw, [128, 128, 4])
    im_blur_raw = tf.decode_raw(features['im_blur_raw'], tf.uint8)
    im_blur_raw = tf.reshape(im_blur_raw, [128, 128, 4])
    im_pan_raw = tf.decode_raw(features['im_pan_raw'], tf.uint8)
    im_pan_raw = tf.reshape(im_pan_raw, [128, 128, 1])

    return im_blur_raw, im_pan_raw, im_mul_raw
def load_examples():
    if a.mode == 'train':
        filename_queue = tf.train.string_input_producer([a.train_tfrecord])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'im_name': tf.FixedLenFeature([], tf.string),
                                           'im_mul_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_raw': tf.FixedLenFeature([], tf.string)
                                       })
    im_mul_raw = tf.decode_raw(features['im_mul_raw'], tf.uint8)
    im_mul_raw = tf.reshape(im_mul_raw, [128, 128, 4])
    im_blur_raw = tf.decode_raw(features['im_blur_raw'], tf.uint8)
    im_blur_raw = tf.reshape(im_blur_raw, [128, 128, 4])
    im_pan_raw = tf.decode_raw(features['im_pan_raw'], tf.uint8)
    im_pan_raw = tf.reshape(im_pan_raw, [128, 128, 1])

    names_batch, inputs1_batch, inputs2_batch, targets_batch = tf.train.shuffle_batch([features['im_name'], im_blur_raw, im_pan_raw, im_mul_raw],
                                                                         batch_size=a.batch_size, capacity=2, )

    steps_per_epoch = int(10 / a.batch_size)
    return Examples(
        imnames=names_batch,
        inputs1=inputs1_batch,
        inputs2=inputs2_batch,
        targets=targets_batch,
        steps_per_epoch=steps_per_epoch,
    )

examples=load_examples()

sess=tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
while(True):
    names = sess.run(examples.imnames)
    print names