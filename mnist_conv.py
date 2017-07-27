import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	with tf.name_scope('input_reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image('input', x_image, 10)

	# We can't initialize these variables to 0 - the network will get stuck.
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	def variable_summaries(var):
		"""
		Attach a lot of summaries to a Tensor (for TensorBoard visualization).
		"""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		variable_summaries(W_conv1)
		b_conv1 = bias_variable([32])
		variable_summaries(b_conv1)
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 32, 64])
		variable_summaries(W_conv2)
		b_conv2 = bias_variable([64])
		variable_summaries(b_conv2)
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

	with tf.name_scope('fully_connected1'):
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
	
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		with tf.name_scope('dropout'):
			keep_prob = tf.placeholder(tf.float32)
			tf.summary.scalar('dropout_keep_probability', keep_prob)
			h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	with tf.name_scope('fully_connected2'):
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	with tf.name_scope('cross_entropy'):
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
				cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	# Merge all the summaries and write them out to
	# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	tf.global_variables_initializer().run()

	# Train the model, and also write summaries.
	# Every 10th step, measure test-set accuracy, and write test summaries
	# All other steps, run train_step on training data, & add training summaries

	def feed_dict(train):
		if train or FLAGS.fake_data:
			xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}

	for i in range(FLAGS.max_steps):
		if i % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		else:  # Record train set summaries, and train
			if i % 100 == 99:  # Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step],
															feed_dict=feed_dict(True),
															options=run_options,
															run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)
			else:  # Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary, i)
	train_writer.close()
	test_writer.close()


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
											default=False,
											help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000,
											help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
											help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
											help='Keep probability for training dropout.')
	parser.add_argument(
			'--data_dir',
			type=str,
			default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
													 'tensorflow/mnist/input_data'),
			help='Directory for storing input data')
	parser.add_argument(
			'--log_dir',
			type=str,
			default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
													 'tensorflow/mnist/logs/mnist_with_summaries'),
			help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
