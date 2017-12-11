import tensorflow as tf
import numpy as np

trainx = np.random.choice(100, 800)
trainy = np.random.choice(100, 800)
#lables = np.zeros([800,1])
index = 1
def getbatch():
	global index
	oneinput = np.ones((1, 6), dtype=np.int32)
	oneout = np.ones((6, 1), dtype=np.int32)
	for step in range(6):
		oneinput[0][step] = trainx[index + step]
		oneout[step] = trainy[index + step]
	index = index + 1
	if index > 794:
		index = 0
	return oneinput, oneout

#inputdata = tf.convert_to_tensor(inputdata, dtype=tf.int32)

train_input = tf.placeholder(dtype=tf.int32, shape=[1,6], name="input")
train_lables = tf.placeholder(dtype=tf.float32, shape=[6, 1], name="labels")

embeddings = tf.get_variable("embeddings", [100, 128], dtype=tf.float32)
embeded = tf.nn.embedding_lookup(embeddings, train_input)

rnn_cell = tf.contrib.rnn.BasicLSTMCell(128)
cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * 3, state_is_tuple=True)

state = cell.zero_state(1, dtype=tf.float32)

output, laststate = tf.nn.dynamic_rnn(cell, embeded, initial_state=state)

output = tf.reshape(output,[6, 128])

weights = tf.Variable(tf.truncated_normal([128, 1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
bias = tf.Variable(tf.random_normal([1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

rnnout = tf.matmul(output, weights) + bias

pre = tf.nn.relu(rnnout)

loss = pre - train_lables
total_loss = tf.reduce_sum(loss)
with tf.name_scope('loss'):
	tf.summary.scalar("all_loss", total_loss)
optime = tf.train.GradientDescentOptimizer(learning_rate=0.1, use_locking=False, name="GradientDescent")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	writer = tf.summary.FileWriter('./path',sess.graph)
	merge = tf.summary.merge_all()
	for x in xrange(1,20):
		inputdata, outputs = getbatch()
		summary, op = sess.run([merge, pre], feed_dict={train_input:inputdata,train_lables:outputs})
		writer.add_summary(summary, x)
	print laststate