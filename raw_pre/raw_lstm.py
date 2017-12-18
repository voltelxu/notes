import tensorflow as tf
import numpy as np
import random

'''
global variable
'''
train_rate = 0.7 #the percent of the train_data set 
lr = 0.5 # the learning rate
batch_size = 2 # the batch size of model
embed_size = 128 # the word embedding size
voc_size = 0 # the vocabulary size
max_num_step = 0 # the max number step unroll of the model
num_layers = 2 # the layers of the model
hidden_size = 500 # the layer size of the model
num_epoch = 1 # the train times of model


#get the vocabulary
def wordid(filename):
	dic = open(filename, 'r')
	voc = dict()
	for line in dic:
		line = line.strip('\n')
		lines = line.split(' ')
		voc[lines[0]] = int(lines[1])
	# lables
	lable = dict()
	lable['Z'] = 0
	lable['B'] = 1
	lable['E'] = 2
	lable['M'] = 3
	lable['S'] = 4
	lable['A'] = 5

	return voc, lable

word_ids, lable_ids = wordid('worddic')
voc_size = len(word_ids)
lable_size = 6

# get data set and chang to ids
# change the data set to train and test data set
def getdata(filename):
	datafile = open(filename, 'r')
	datas = list()
	#the max length of the sentence
	max_scelen = 0

	for line in datafile:
		words = line.strip('\n').split(" ")
		tup = list()
		data = list()
		lable = list()
		for word in words:
			terms = word.split(":")
			data.append(word_ids.get(terms[0], 0))
			lable.append(lable_ids[terms[1]])
			if len(data) > max_scelen:
				max_scelen = len(data)
		tup.append(data)
		tup.append(lable)
		datas.append(tup)

	len_data = len(datas)
	#shuffle the data set
	random.shuffle(datas)
	#split position
	split = int(len_data * train_rate)
	#split dataset to train and test sets
	train = datas[0 : split]
	test = datas[split : len_data]
	#train_ data
	train_data = [t for t in [x[0] for x in train]]
	#train_lable
	train_lable = [t for t in [x[1] for x in train]]
	#test_data
	test_data = [t for t in [x[0] for x in test]]
	#test_lable
	test_lable = [t for t in [x[1] for x in test]]
	#return train and test set
	return train_data, train_lable, test_data, test_lable, max_scelen

train_data,train_lable, test_data, test_lable, max_len = getdata("test")
# max sequence length 
max_num_step = max_len

# get batch input data
def getbatch(data, lable, start, batch_size):
	batch_data = data[start : start + batch_size]
	batch_seqlen = [length for length in [len(x) for x in batch_data]]

	batch_datas = list()
	for d in batch_data:
		l = len(d)
		d += [0 for i in range(max_num_step - l)]
		batch_datas.append(d)

	batch_lable = lable[start : start + batch_size]
	batch_lables = list()
	for b in batch_lable:
		l = len(b)
		b += [0 for i in range(max_num_step - l)]
		batch_lables.append(b)
	
	return batch_datas, batch_lables, batch_seqlen

# batch_data, batch_lable, batch_seqlen = getbatch(train_data, train_lable, 1, 8)
# print len(batch_data)

def dynamicRNN(inputs, seqlen, weights):
	#
	inputs = tf.unstack(inputs, max_num_step, 1)
	#lstm cell
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	#outputs and state
	outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32, 
		sequence_length=seqlen)
	# outputs shape [num_step, batch_size, hidden_size]
	outputs = tf.stack(outputs)
	# outputs shape [batch_size, num_step, hidden_size]
	outputs = tf.transpose(outputs, [1, 0, 2])

	outputs = tf.reshape(outputs, [-1, hidden_size])
	return tf.matmul(outputs, weights)

def model():
	# input data
	input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_num_step])
	# targets lables
	target = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_num_step])
	# true seq lenght
	seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
	# embeddings
	embedding = tf.get_variable(name='embedding', shape=[voc_size, embed_size])
	# change word to tensor
	inputs = tf.nn.embedding_lookup(embedding, input_data)
	# weights 
	weights = tf.get_variable("weights",[hidden_size, lable_size])
	# rnn cell
	pred = dynamicRNN(inputs, seqlen, weights)
	# 
	logits = tf.reshape(pred, [batch_size, max_num_step, lable_size])

	loss = tf.contrib.seq2seq.sequence_loss(logits,
	 			target, 
	 			tf.ones([batch_size, max_num_step]))
	tf.summary.scalar("loss", loss)
	cost = tf.reduce_mean(loss)

	softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, lable_size]))
	predict = tf.cast(tf.argmax(softmax_out, axis=1),tf.int32)
	correct = tf.equal(predict, tf.reshape(target,[-1]))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar("accuracy", accuracy)

	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	opt = tf.train.GradientDescentOptimizer(learning_rate=lr).apply_gradients(zip(grads,tvars))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		writer = tf.summary.FileWriter('./path', sess.graph)
		merged = tf.summary.merge_all()
		# train
		epoch_size = len(train_data)/batch_size
		testsize = len(test_data)/batch_size
		for step in range(20000):
			for i in xrange(0, epoch_size, 1):
				x, y, l = getbatch(train_data, train_lable, i * batch_size, batch_size)
				sess.run([opt], feed_dict={input_data : x, target : y, seqlen : l})
			sum_loss = 0.0
			sum_ac = 0.0
			num = 0
			for j in xrange(0, testsize, 1):
				num = num + 1
				tx, ty, tl = getbatch(test_data, test_lable, j * batch_size, batch_size)
				summary, ac, loss = sess.run([merged, accuracy, cost], feed_dict={input_data : tx, target : ty, seqlen : tl})
				sum_ac = sum_ac + ac
				sum_loss = sum_loss + loss
			sum_ac = sum_ac/num
			if step % 100 == 0:
				print "step "+str(step)+", accuracy "+str(sum_ac)+", loss " + str(sum_loss)
model()