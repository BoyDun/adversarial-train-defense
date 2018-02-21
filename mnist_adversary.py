import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
batch_size = 128
n_epochs = 10
alpha = 0.5
LAYER_1 = 512
LAYER_2 = 256
LAYER_3 = 128
INPUT = 784
OUTPUT = 10

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

y_ = tf.placeholder(tf.float32, [None, OUTPUT])
x_norm = tf.placeholder(tf.float32, [None, INPUT])
x_adv = tf.placeholder(tf.float32, [None, INPUT])

W_fc1 = weight_variable([INPUT, LAYER_1])
W_fc2 = weight_variable([LAYER_1, LAYER_2])
W_fc3 = weight_variable([LAYER_2, LAYER_3])
W_fc4 = weight_variable([LAYER_3, OUTPUT])

b_fc1 = bias_variable([LAYER_1])
b_fc2 = bias_variable([LAYER_2])
b_fc3 = bias_variable([LAYER_3])
b_fc4 = bias_variable([LAYER_4])

# Regular examples
h_fc1_norm = tf.nn.leaky_relu(tf.matmul(x_norm, W_fc1) + b_fc1, alpha=0.1)
h_fc2_norm = tf.nn.leaky_relu(tf.matmul(h_fc1_norm, W_fc2) + b_fc2, alpha=0.1)
h_fc3_norm = tf.nn.leaky_relu(tf.matmul(h_fc2_norm, W_fc3) + b_fc3, alpha=0.1)
final_norm = tf.matmul(h_fc3_norm, W_fc4) + b_fc4

# Adversarial examples
h_fc1_adv = tf.nn.leaky_relu(tf.matmul(x_adv, W_fc1) + b_fc1, alpha=0.1)
h_fc2_adv = tf.nn.leaky_relu(tf.matmul(h_fc1_adv, W_fc2) + b_fc2, alpha=0.1)
h_fc3_adv = tf.nn.leaky_relu(tf.matmul(h_fc2_adv, W_fc3) + b_fc3, alpha=0.1)
final_adv = tf.matmul(h_fc3_adv, W_fc4) + b_fc4

loss = -alpha * tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=final_norm) - \
       (1 - alpha) * tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=final_adv)

