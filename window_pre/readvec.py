import numpy as np
import tensorflow as tf
import collections
import zipfile

def readvecfile(filename):
	file = open(filename, 'r')
	wordvec = dict()
	wordindex = dict()
	for line in file:
		line = line.strip(' \n')
		data = line.split(":")
		word = data[0]
		vec = data[1].split(" ")
		vec = map(eval, vec)
		vec = np.array(vec)
		wordvec[word] = vec
		wordindex[word] = len(wordindex)
	return wordvec, wordindex

wordvec, wordindex = readvecfile("true_embed")
print(wordindex)

def readdata(filename, n_word):
	file = open(filename, 'r')
	text = ""
	for l in file:
		text = text + l.strip('\n') + " "
	text = text.split(" ")
	count = [['UNK', -1]]
	count.extend(collections.Counter(text).most_common(n=n_word))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in text:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = readdata('english', 100)

def getembedings(dictionary, wordvec):
	embeddings = np.zeros(((len(dictionary),128)))
	for word in dictionary:
		#print(word)
		if word in wordvec:
			print(word)
			embeddings[dictionary[word]] = wordvec[word]
	return embeddings

embeddings = getembedings(dictionary, wordvec)
#print(embeddings)
em = tf.Variable(initial_value=embeddings)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	init.run()
#	print(em[1].eval())

