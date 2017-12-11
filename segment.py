import tensorflow as tf
import numpy as np
import os
import collections
import zipfile

word_size = 150
embedding_size = 128

def readvecfile(filename):
	file = open(filename, 'r')
	embeddings = np.zeros((word_size + 1, embedding_size))
	dictionary = dict()
	for line in file:
		line = line.strip(' \n')
		data = line.split(":")
		word = data[0]
		vec = data[1].split(" ")
		vec = map(eval, vec)
		vec = np.array(vec)
		embeddings[len(dictionary)] = vec
		dictionary[word] = len(dictionary)
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return embeddings, dictionary, reversed_dictionary

embeddings, dictionary, reversed_dictionary= readvecfile("true_embed")

def build_dataset(filename):
	file = open(filename, 'r')
	words = ""
	lables = ""
	for line in file:
		line = line.strip('\n')
		line = line.split(" ")
		words = words + line[0] + " "
		lables = lables + line[1] + " "
	words = words.split(" ")
	lables = lables.split(" ")
	data = list()
	for word in words:
		index = dictionary.get(word, 0)
		data.append(index)
	return data, lables

data, data_y = build_dataset('lables')
data_index = 0

def get_batchset(batch_size, skip_window):
	global data_index
	span = skip_window * 2 + 1
	batch = np.ndarray(shape=(batch_size, span), dtype=np.int32)
	lables = np.ndarray(shape=(batch_size,4), dtype=np.int32)
	oldlables =  np.ndarray(shape=(batch_size), dtype=np.int32)
	if data_index + span > len(data):
		data_index = 0
	if data_index + batch_size > len(data):
		data_index = 0
	for i in range(batch_size):
		one_data = np.ndarray(shape=(span), dtype=np.int32)
		one_data[skip_window] = data[data_index]
		start = data_index - skip_window
		for j in range(span):
			one_data[j] = data[j + start]
		batch[i] = one_data
		y = data_y[data_index]
		yold = data_y[data_index - 1]
		if y == "B":
			lables[i] = np.asarray((1,0,0,0))
			oldlables[i] = 0
		if y == "E":
			lables[i] = np.asarray((0,1,0,0))
			oldlables[i] = 1
		if y == "M":
			lables[i] = np.asarray((0,0,1,0))
			oldlables[i] = 2
		if y == "S":
			lables[i] = np.asarray((0,0,0,1))
			oldlables[i] = 3
		data_index = data_index + 1
	return batch, lables, oldlables

def get_validset(index, batch_size, skip_window):
	span = skip_window * 2 + 1
	batch = np.ndarray(shape=(batch_size, span), dtype=np.int32)
	lables = np.ndarray(shape=(batch_size,4), dtype=np.int32)
	oldlables =  np.ndarray(shape=(batch_size), dtype=np.int32)
	if index + span > len(data):
		index = 0
	if index + batch_size > len(data):
		index = 0
	for i in range(batch_size):
		one_data = np.ndarray(shape=(span), dtype=np.int32)
		one_data[skip_window] = data[index]
		start = index - skip_window
		for j in range(span):
			one_data[j] = data[j + start]
		batch[i] = one_data
		y = data_y[index]
		yold = data_y[index - 1]
		if y == "B":
			lables[i] = np.asarray((1,0,0,0))
			oldlables[i] = 0
		if y == "E":
			lables[i] = np.asarray((0,1,0,0))
			oldlables[i] = 1
		if y == "M":
			lables[i] = np.asarray((0,0,1,0))
			oldlables[i] = 2
		if y == "S":
			lables[i] = np.asarray((0,0,0,1))
			oldlables[i] = 3
		index = index + 1
	return batch, lables, oldlables

batch_size = 8
skip_window = 2
span = skip_window *2 + 1
num_node = 128

valid_input, valid_out,valid_oldl = get_validset(588, batch_size, skip_window)
# print valid_input

graph = tf.Graph()
with graph.as_default():
	#input data
	train_input = tf.placeholder(tf.int32, shape=[batch_size, span], name="X")
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 4], name="Y")
	train_oldlables = tf.placeholder(tf.int32, shape=[batch_size], name=None)

	embeddings = tf.Variable(initial_value=embeddings, trainable=True, dtype=tf.float32)
	lookupembed = tf.nn.embedding_lookup(embeddings, train_input)
	embed = tf.reshape(lookupembed, shape=[batch_size, embedding_size * span])
	#transpose
	transpose = tf.Variable(tf.random_normal([4, 4], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
	trans = tf.nn.embedding_lookup(transpose, train_oldlables)
	#hidden one
	W1 = tf.Variable(tf.random_normal([embedding_size * span, num_node],stddev=0.6), name="w1")
	b1 = tf.Variable(tf.random_normal([num_node], stddev=0.35), name="b1")
	d1 = tf.matmul(embed, W1) + b1
	y1 = tf.nn.sigmoid(d1, name="hidout_1")
	#hidden two
	w2 = tf.Variable(tf.random_normal([num_node, 4], stddev=0.6), name="w2")
	b2 = tf.Variable(tf.random_normal([4], stddev=0.35), name="b1")
	d2 = tf.matmul(y1, w2) + b2
	y2 = tf.nn.sigmoid(d2, name="hidout_2")
	y3 = tf.multiply(trans, y2)
	#output
	y = tf.nn.softmax(y3, name="output")

	loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=y)
	reduce_loss = tf.reduce_mean(loss)
	tf.summary.scalar("loss", reduce_loss)
	optimizer = tf.train.GradientDescentOptimizer(0.1, name="GradientDescent").minimize(reduce_loss)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		init.run()
		writer = tf.summary.FileWriter('./path', sess.graph)
		merged = tf.summary.merge_all()
		for step in range(1000):
			x, l, oldl= get_batchset(batch_size, skip_window)
			summary, sum_loss, _, y = sess.run([merged, reduce_loss, optimizer, y2], feed_dict={train_input:x, train_labels:l, train_oldlables:oldl})
			writer.add_summary(summary, global_step=step)
		pre = sess.run(y2, feed_dict={train_input:valid_input,train_oldlables:valid_oldl})
		cru = 0
		for v in range(1000):
			va = 0
			for line in range(len(pre)):
				linedata = pre[line]
				maxd = max(linedata)
				pos = np.argwhere(linedata == maxd)
				if valid_out[line][pos] == 1:
			 		va = va + 1
			if va > cru:
				cru = va
		print cru