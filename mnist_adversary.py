import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fast_gradient

epochs = 120
learning_rate = 0.01
batch_size = 100
dropout = 0.5
n_epochs = 10
alpha = 0.5
beta = 1
LAYER_1 = 512
LAYER_2 = 256
LAYER_3 = 128
INPUT = 784
OUTPUT = 10

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float64, stddev=0.01)
    return tf.Variable(initial, dtype=tf.float64)

def bias_variable(shape):
    initial = tf.constant(0.0, dtype=tf.float64, shape=shape)
    return tf.Variable(initial, dtype=tf.float64)

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

y_ = tf.placeholder(tf.float64, [None, OUTPUT])
x_norm = tf.placeholder(tf.float64, [None, INPUT])
x_adv = tf.placeholder(tf.float64, [None, INPUT])

W_fc1 = weight_variable([INPUT, LAYER_1])
W_fc2 = weight_variable([LAYER_1, LAYER_2])
W_fc3 = weight_variable([LAYER_2, LAYER_3])
W_fc4 = weight_variable([LAYER_3, OUTPUT])

b_fc1 = bias_variable([LAYER_1])
b_fc2 = bias_variable([LAYER_2])
b_fc3 = bias_variable([LAYER_3])
b_fc4 = bias_variable([OUTPUT])

# Regular examples
h_fc1_norm = tf.nn.relu(tf.matmul(x_norm, W_fc1) + b_fc1)
h_fc2_norm = tf.nn.relu(tf.matmul(h_fc1_norm, W_fc2) + b_fc2)
h_fc3_norm = tf.nn.relu(tf.matmul(h_fc2_norm, W_fc3) + b_fc3)
final_norm = tf.matmul(h_fc3_norm, W_fc4) + b_fc4
cross_norm = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_norm) 

# Adversarial examples
h_fc1_adv = tf.nn.relu(tf.matmul(x_adv, W_fc1) + b_fc1)
h_fc2_adv = tf.nn.relu(tf.matmul(h_fc1_adv, W_fc2) + b_fc2)
h_fc3_adv = tf.nn.relu(tf.matmul(h_fc2_adv, W_fc3) + b_fc3)
final_adv = tf.matmul(h_fc3_adv, W_fc4) + b_fc4
cross_adv = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_adv)

loss = -alpha * cross_norm - (1 - alpha) * cross_adv

# For generating adversarial examples
sm_norm = tf.nn.softmax(final_norm)

# Discriminator
W_d1 = weight_variable([LAYER_2, LAYER_3])
W_d2 = weight_variable([LAYER_3, 2])
b_d1 = bias_variable([LAYER_3])
b_d2 = bias_variable([2])
keep_prob_input = tf.placeholder(tf.float64)
drop_reg_discr = tf.nn.dropout(h_fc2_norm, keep_prob=keep_prob_input)
drop_adv_discr = tf.nn.dropout(h_fc2_adv, keep_prob=keep_prob_input)
discr_norm1 = tf.nn.relu(tf.matmul(drop_reg_discr, W_d1) + b_d1)
discr_adv1 = tf.nn.relu(tf.matmul(drop_adv_discr, W_d1) + b_d1)
final_discr_norm = tf.matmul(discr_norm1, W_d2) + b_d2
final_discr_adv = tf.matmul(discr_adv1, W_d2) + b_d2
cross_discr_norm = tf.nn.softmax_cross_entropy_with_logits(labels=np.asarray([1,0]), logits=final_discr_norm)
cross_discr_adv = tf.nn.softmax_cross_entropy_with_logits(labels=np.asarray([0,1]), logits=final_discr_adv)
discr_loss = - (cross_discr_norm + cross_discr_adv)

# define training step and accuracy
#print loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(final_norm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

train_step_discr = tf.train.AdamOptimizer(learning_rate).minimize(discr_loss)
correct_norm_discr = tf.equal(0, tf.argmax(cross_discr_norm,1))
correct_adv_discr = tf.equal(1, tf.argmax(cross_discr_adv,1))
norm_acc = tf.reduce_mean(tf.cast(correct_norm_discr, tf.float64))
adv_acc = tf.reduce_mean(tf.cast(correct_adv_discr, tf.float64))
comb_acc = (norm_acc + adv_acc) / 2
# create a saver
saver = tf.train.Saver()

# initialize graph
init = tf.initialize_all_variables()

# generating adversarial images
fgm_eps = tf.placeholder(tf.float64, ())
fgm_epochs = tf.placeholder(tf.float64, ())
adv_examples = fast_gradient.fgmt(x_norm, final_norm, sm_norm, y=y_, eps=fgm_eps, epochs=fgm_epochs) 

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        for j in range(mnist.train.num_examples/batch_size):
            input_images, correct_predictions = mnist.train.next_batch(batch_size)
            final_logits = sess.run(final_norm, feed_dict={x_norm: input_images})
            final_output = sess.run(sm_norm, feed_dict={x_norm: input_images})
            adv_images = sess.run(adv_examples, feed_dict={x_norm: input_images, final_norm: final_logits, sm_norm: final_output, y_:correct_predictions, fgm_eps: 0.01, fgm_epochs: 1}) 
            #GENERATE ADVERSARIAL IMAGES
            if j == 0:
                discr_accuracy = sess.run(comb_acc, feed_dict={keep_prob_input:1.0, x_norm:input_images, x_adv:adv_images, y_:correct_predictions})
                train_accuracy = sess.run(accuracy, feed_dict={x_norm:input_images, x_adv:adv_images, y_:correct_predictions})
                path = saver.save(sess, 'mnist_save')
            sess.run(train_step_discr, feed_dict={keep_prob_input:dropout, x:input_images, x_adv:adv_images, y_:correct_predictions}) 
            sess.run(train_step, feed_dict={x:input_images, x_adv:adv_images, y_:correct_predictions})



