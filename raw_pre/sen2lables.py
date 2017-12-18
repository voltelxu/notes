import re

def word2lables(filename, outfile):
	file = open(filename, 'r')
	out = open(outfile, 'w+')
	k = 0
	for line in file:
		line = line.strip('\n')
		line = line.strip('\r\n')
		line = line.strip(' ')
		data = ""
		if len(line) <= 3:
			continue
		words = line.split(" ")
		k = k + 1
		for word in words:
			if len(word) < 3:
				continue
			elif len(word) == 3:
				if word == '<s>':
					data = data + word + ":A "
				elif word == '<e>':
					data = data + word + ":Z "
				else:
					data = data + word + ":S "
				continue
			elif len(word) > 3:
				for i in range(0, len(word), 3):
					if i == 0:
						data = data + word[i:3] + ":B "
					elif i == len(word) - 3:
						data = data + word[i:i+3] + ":E "
					else:
						data = data +word[i:i+3] + ":M "
		out.write(data.strip(" ")+"\n")
	out.close()
	file.close()

word2lables('raw', 'lables')