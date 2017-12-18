# -*- coding:utf-8 -*-
import re

def parse(line):
	res = ""
	sen = line.split('. ')
	for sub in sen:
		subsen = sub.split(', ')
		for sentence in subsen:
			sentence = str(sentence)
			sentence = unicode(sentence, "utf-8")
			sentence = re.sub(u'[^\uAC00-\uD7A3 ]', '', sentence)
			sentence = sentence.encode('utf-8')
			sentence = re.sub('[ ]+', ' ', sentence)
			sentence = sentence.strip(" ")
			if sentence == '' or len(sentence) <=12:
				res = res + ""
			else:
				res = res + "<s> " + sentence + " <e>\n"
	return res


def sen2raw(infile, outfile):
	file = open(infile, 'r')
	out = open(outfile, 'w')
	m_len = 0
	len_s = ""
	for line in file:
		line = line.strip('\n')
		line = line.strip('\r\n')
		length = len(line)
		sentence = ""
		if length < 9:
			continue
		else:
			sentence = parse(line)
		ll = sentence.split("\n")
		for x in ll:
			x = x.replace(" ", '')
			tmp = len(x)
			if tmp > m_len:
				m_len = tmp
				len_s = x
		out.write(sentence)
	print m_len
	print len_s
	out.close()
	file.close()

sen2raw('out', 'raw')