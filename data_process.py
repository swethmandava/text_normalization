import csv
from collections import defaultdict
import numpy as np

#Create a character lookup table
extended_ascii = [chr(i) for i in xrange(256)]
a =  (extended_ascii.index('A'))
b =  (extended_ascii.index('Z'))

#Remove uppercase characters
extended_ascii[a:b] = []
vocabulary = defaultdict(int)
cnt = 1
for k in extended_ascii:
	vocabulary[k] = cnt
	cnt = cnt + 1 

num =  vocabulary['\x00']

#Create a word encoding table
cnt = 1
f = open('try.csv', 'rb')
reader = csv.reader(f)
word_encoding = defaultdict(int)
j = 1

for row in reader:
	if j == 1:
		j = -1
		continue
	if ' ' in row[4]:
		a = row[4].split(' ')
		for s in a:
			if s not in word_encoding:
				word_encoding[s] = cnt
				print cnt
				cnt = cnt + 1
print 'SELF', cnt
word_encoding['SELF'] = cnt
cnt = cnt + 1
print 'Null', cnt
word_encoding['\x00'] = cnt
cnt = cnt + 1
print 'END', cnt
word_encoding['END'] = cnt
f.close()

inverse_dict = dict(zip(word_encoding.values(), word_encoding.keys()))

f = open('try.csv', 'rb')
reader = csv.reader(f)
data = []
btch = 5
i = 1
max_len = 500
batch = np.full([btch,max_len],num)
window = 3

sid = '0'
i = 1
words = []
lst = []
lst1 = []
cnt = 0
c = 0
c1 = 0
num1 = word_encoding['\x00']

print num1
c2 = 0
c3 = 0

x_length = np.zeros([btch,1])
batch_y = np.full([btch,max_len],num1)
b_y = np.full([500,max_len],num1)
y_length = np.zeros([btch,1])

for row in reader:
	c2 = 0
	if i == 1:
		i = -1
		continue	
	if sid == row[0]:
		print row[3]
		words.append(row[3].lower())
		a = 1

		if row[3] == row[4]:
			b_y[c3,c2]= (word_encoding['SELF'])
			b_y[c3,c2+1] = (word_encoding['END'])
			c3 = c3 + 1 
		else:
			w = row[4].split(' ')
			for x in w:
				b_y[c3,c2]= (word_encoding[x])
				c2 = c2 + 1
			b_y[c3,c2+1] = (word_encoding['END'])	
			c3 = c3 + 1

	else:
		num_words = len(words)
		for word_ind in range(window,num_words - window):
			words_batch = [words[word_ind-3], words[word_ind-2], words[word_ind-1], words[word_ind],words[word_ind+1],words[word_ind+2],words[word_ind+3]]
			batch_y[cnt,:] = b_y[word_ind,:]

			for wrd in words_batch:
				for ind in range(len(wrd)):
					batch[cnt,c] = vocabulary[wrd[ind]]
					c = c + 1
					if wrd[ind] != '.':
						batch[cnt,c] = vocabulary[' ']
						c = c + 1

			y_length[cnt] = c1			
			x_length[cnt] = c			
			cnt = cnt + 1
			c = 0
			c1 = 0
			if cnt == btch :
				print x_length
				print batch
				print batch_y
				c1 = 0
				batch_y = np.full([btch,max_len],num1)
				x_length = np.zeros([btch,1])
				batch = np.full([btch,max_len],num)
				cnt = 0
				c = 0
		
		b_y = np.full([500,max_len],num1)
		words_batch = []	
		words = []
		c3 = 0
		c2 = 0
		sid = row[0]	
		print 'NOW the sid is:' , row[0]
		print 'First word to be appended is', row[3]
		
		words.append(row[3].lower()) 

		print 'Checking the updation of' , row[3] 
		
		if row[3] == row[4]:
			b_y[c3,c2]= (word_encoding['SELF'])
			b_y[c3,c2+1] = (word_encoding['END'])
			c3 = c3 + 1 
		else:
			print 'the word being checked is',row[3]
			w = row[4].split(' ')
			for x in w:
				print word_encoding[x]
				b_y[c3,c2]= (word_encoding[x])
				c2 = c2 + 1
			b_y[c3,c2+1] = (word_encoding['END'])	
			c3 = c3 + 1
		a = 0



num_words = len(words)
for word_ind in range(window,num_words - window):
		
		print [words[word_ind-3], words[word_ind-2], words[word_ind-1], words[word_ind],words[word_ind+1],words[word_ind+2],words[word_ind+3]]
		words_batch =[words[word_ind-3], words[word_ind-2], words[word_ind-1], words[word_ind],words[word_ind+1],words[word_ind+2],words[word_ind+3]]
		print '******'
		print b_y[word_ind,:]
		print '******'
		batch_y[cnt,:] = b_y[word_ind,:]
		for wrd in words_batch:
			for ind in range(len(wrd)):
				batch[cnt,c] = vocabulary[wrd[ind]]
				c = c + 1
				if wrd[ind] != '.':
					batch[cnt,c] = vocabulary[' ']
					c = c + 1
		y_length[cnt] = c1			
		x_length[cnt] = c	
		#cnt = cnt + 1
		c = 0
		if cnt == btch :
			# print batch_y
			# print batch
			# print x_length
			#print batch_y
			a = 1
# print batch_y			
		
# # print lst1
# # print vocabulary['r']