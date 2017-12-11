import tensorflow as tf
import numpy as np
import collections
import os

def read_words(filename):
	with tf.gfile.GFile(filename, mode='r') as f:
		line = f.read().replace('\n', ' ')
		line = line.split(' ')
		data = [line[i] for i in xrange(0, len(line) - 1, 2)]
		lables = [line[i] for i in xrange(1, len(line), 2)]
		return data, lables
data, lables = read_words('lables')

# print len(data), len(lables)

counter_data = collections.Counter(data)
counter_lables = collections.Counter(lables)
count_pairs_data = sorted(counter_data.items(), key=lambda x:(-x[1],x[0]))
count_pairs_lables = sorted(counter_lables.items(), key=lambda x:(-x[1],x[0]))

words, _ = list(zip(*count_pairs_data))
lable, _ = list(zip(*count_pairs_lables))
word_to_id = dict(zip(words, range(len(words))))
lable_to_id = dict(zip(lable, range(len(lable))))

data = [word_to_id[word] for word in data if word in word_to_id]
lables = [lable_to_id[l] for l in lables if l in lable_to_id]

vocab_size = len(words)
lable_size = len(lable)
enbed_size = 64
batch_size = 2
num_steps = 25
num_layers = 2
hidden_size = 128

# raw_data = tf.convert_to_tensor(data, dtype=tf.int32)
raw_data = np.asarray(data)
raw_lable = np.asarray(lables)
data_len = len(raw_data)
lable_len = len(raw_lable)
batch_len = data_len//batch_size
data = np.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])
lables = np.reshape(raw_lable[0:batch_size * batch_len], [batch_size, batch_len])
epoch_size = (batch_len - 1) // num_steps

inputdata = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps], name="inputdata")
target = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps], name="target")

# embedding = tf.Variable(tf.random_uniform([vocab_size, enbed_size], -0.05, 0.05))
embedding = tf.get_variable("embeddings", [vocab_size, enbed_size], dtype=tf.float32)
inputs = tf.nn.embedding_lookup(embedding, inputdata)

cell1 = tf.contrib.rnn.BasicLSTMCell(hidden_size)
# cell1o = tf.contrib.rnn.MultiRNNCell([cell1]*num_layers, state_is_tuple=True)

cell2 = tf.contrib.rnn.BasicLSTMCell(hidden_size * 2)

cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)

init_state = cell.zero_state(batch_size, dtype=tf.float32)

'''state = init_state
output = list()
with tf.variable_scope('RNN'):
	for time in range(num_steps):
		if time > 0:
			tf.get_variable_scope().reuse_variables()
		(cellout, state) = cell(inputs[:, time,:],state)
		output.append(cellout)
'''

output, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)

output = tf.reshape(output, [batch_size*num_steps, hidden_size * 2])
softmaxw = tf.Variable(tf.random_normal([hidden_size *2, lable_size], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
softmaxb = tf.Variable(tf.random_normal([lable_size], mean=0.0, stddev=0.5, dtype=tf.float32, seed=None, name=None))
logitss = tf.nn.xw_plus_b(output, softmaxw, softmaxb, name=None)

logits = tf.reshape(logitss, [batch_size, num_steps, lable_size], name=None)
loss = tf.contrib.seq2seq.sequence_loss(logits,target,tf.ones([batch_size,num_steps],dtype=tf.float32),average_across_timesteps=False,average_across_batch=True)
cost = tf.reduce_sum(loss)

softmaxout = tf.nn.softmax(tf.reshape(logits, [-1, lable_size]))
predict = tf.cast(tf.argmax(softmaxout, axis=1),tf.int32)
correct = tf.equal(predict, tf.reshape(target, [-1]), name=None)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.5).apply_gradients(zip(grads,tvars))

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter('./path')

	max_cur = 0
	writer.add_graph(sess.graph)
	for i in range(200):
		for i in range(epoch_size-1):
			x = data[:, i * num_steps:(i + 1) * num_steps]
			y = lables[:, i * num_steps :(i + 1) * num_steps]
			out = sess.run(opt, feed_dict={inputdata:x,target:y})
		m = epoch_size - 1
		vin = data[:, m * num_steps:(m + 1) * num_steps]
		vou = lables[:, m * num_steps:(m + 1) * num_steps]
		ac = sess.run(predict, feed_dict={inputdata:vin})
		ac = np.reshape(ac,(batch_size, num_steps))
		cur = 0
		li = ""
		length = len(ac)
		# for r in ac:
		for r in range(length):
			for c in range(num_steps):
				if ac[r][c] == vou[r][c]:
					cur = cur + 1
			# for rr in r:
				# li = li + lable[rr]
		if cur > max_cur:
			max_cur = cur
	print max_cur