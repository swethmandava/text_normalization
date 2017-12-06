import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math
import matplotlib as mlp
mlp.use("Agg")
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import sys

path = ''
data_exists = 1
USE_GPU = 0 
is_train = True 

def initialize_all_variables(sess=None):
	# Credit : yaroslavvb
	"""Initializes all uninitialized variables in correct order. Initializers
	are only run for uninitialized variables, so it's safe to run this multiple
	times.
	Args:
	  sess: session to use. Use default session if None.
	"""

	from tensorflow.contrib import graph_editor as ge
	def make_initializer(var): 
		def f():
			return tf.assign(var, var.initial_value).op
		return f

	def make_noop(): return tf.no_op()

	def make_safe_initializer(var):
		"""Returns initializer op that only runs for uninitialized ops."""
		return tf.cond(tf.is_variable_initialized(var), make_noop,
			make_initializer(var), name="safe_init_"+var.op.name).op

	if not sess:
		sess = tf.get_default_session()
	g = tf.get_default_graph()
	  
	safe_initializers = {}
	for v in tf.all_variables():
		safe_initializers[v.op.name] = make_safe_initializer(v)
	  
	# initializers access variable vaue through read-only value cached in
	# <varname>/read, so add control dependency to trigger safe_initializer
	# on read access
	for v in tf.all_variables():
		var_name = v.op.name
		var_cache = g.get_operation_by_name(var_name+"/read")
		ge.reroute.add_control_inputs(var_cache, [safe_initializers[var_name]])

	sess.run(tf.group(*safe_initializers.values()))

	# remove initializer dependencies to avoid slowing down future variable reads
	for v in tf.all_variables():
		var_name = v.op.name
		var_cache = g.get_operation_by_name(var_name+"/read")
		ge.reroute.remove_control_inputs(var_cache, [safe_initializers[var_name]]) 

def create_dictionary(file,btch,max_len):

	# Create a character lookup table
	extended_ascii = [chr(i) for i in xrange(256)]

	# Remove uppercase characters
	first =  (extended_ascii.index('A'))
	second =  (extended_ascii.index('Z'))
	extended_ascii[first:second] = []

	# Create a dictionary of all the characters
	vocabulary = defaultdict(int)
	cnt = 1
	for k in extended_ascii:
		vocabulary[k] = cnt
		cnt = cnt + 1 

	# Classes for the data, except for plain and punct everything is changed
	classes = ['ORDINAL', 'DIGIT', 'LETTERS', 'VERBATIM', 'ADDRESS', 'DECIMAL', 'TIME', 'MONEY', 'TELEPHONE', 'CARDINAL', 'FRACTION', 'MEASURE', 'DATE', 'ELECTRONIC']
	classes_unchanged = ['PLAIN','PUNCT']
	# Create a word encoding table
	cnt = 1
	f = open(file, 'rb')
	reader = csv.reader(f)
	word_index = defaultdict(int)
	frs = 1
	a = []

	for row in reader:
		# First line of file is description, so skip
		if frs == 1:
			frs = -1
			continue
		# Check is word has to be normalized, that is their class is not plain or punct	
		if row[2] in classes:
			a = row[4].split(' ')
			# Words get their own embeddings, only the ones that have to be changed
			for s in a:
				if s not in word_index:
					word_index[s] = cnt
					cnt = cnt + 1		
	f.close()
	# All the other words get encoded as SELF and SIL, and also END and NULL
	# Add those to the word_index table
	word_index['SELF'] = cnt
	cnt = cnt + 1
	word_index['SIL'] = cnt
	cnt = cnt + 1
	word_index['\x00'] = cnt
	cnt = cnt + 1
	word_index['END'] = cnt

	return word_index

	
def BiRNN(num_hidden, num_classes, num_input, num_groups, learning_rate, encoding_layers, vocab_size,
	decoding_layers, start_token, end_token):

	
	# Inputs
	#max_in_time --> encoder time steps
	#max_out_time --> decoder time steps
	
	embedding_encoder = tf.Variable(tf.random_normal((num_classes, num_input)))
	class_encoder = tf.Variable(tf.random_normal((num_groups, num_hidden)))
	decoding_encoder = tf.one_hot(range(vocab_size), vocab_size, dtype=tf.float32) 
	
	x_input = tf.placeholder(tf.int32, [None, None])
	X_length = tf.placeholder(tf.int32, [None])
	y_input = tf.placeholder(tf.int32, [None, None])
	y_shifted_input = tf.placeholder(tf.int32, [None, None])
	groups = tf.placeholder(tf.int32, [None])

	X = tf.nn.embedding_lookup(embedding_encoder, x_input)
	Y = tf.nn.embedding_lookup(decoding_encoder, y_input)
	#Makes batch size x num hidden to batch size x 1 x num_hidden
	#Appropriate for attention
	groups_embedding = tf.expand_dims(tf.nn.embedding_lookup(class_encoder, groups), 1)

	start_token = tf.nn.embedding_lookup(decoding_encoder, start_token)
	end_token = tf.nn.embedding_lookup(decoding_encoder, end_token)

	Y_length = tf.placeholder(tf.int32, [None])
	 
	batch_size = tf.shape(X)[0]
	max_in_time = tf.shape(x_input)[1]
	max_out_time = tf.shape(y_input)[1]
	num_gpus =3
	
	if USE_GPU :
		
		cells = []
		for i in range(encoding_layers):
			cells.append(tf.contrib.rnn.DeviceWrapper(
					tf.nn.rnn_cell.LSTMCell(num_hidden),
					"/gpu:%d" % (2))) #     encoding_layers % num_gpus)))
		
		lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_hidden, 
			groups_embedding)
		lstm_fw_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_fw_cell, 
			attention_mechanism, attention_layer_size=num_hidden)		
		
		lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_hidden, 
			groups_embedding)
		lstm_bw_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
			attention_mechanism, attention_layer_size=2*num_hidden)
		
	else :    

		# Forward direction stacked lstm cell
		lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_hidden) for _ in range(encoding_layers)])
		# Backward direction stacked lstm cell
		lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_hidden) for _ in range(encoding_layers)])

	((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state)) = \
	tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X, sequence_length=X_length, dtype=tf.float32)
	
	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 1)
	# batch size x 2num_hidden 

	# UNcomment if we need state and modify basic decoder accordingly
	# if isinstance(encoder_fw_state, rnn.LSTMStateTuple):  # LstmCell
	# 	state_c = tf.concat(
	# 		(encoder_fw_state.c, encoder_bw_state.c), 1)
	# 	state_h = tf.concat(
	# 		(encoder_fw_state.h, encoder_bw_state.h), 1)
	# 	encoder_state = rnn.LSTMStateTuple(c=state_c, h=state_h)
	# elif isinstance(encoder_fw_state, tuple) \
	# 		and isinstance(encoder_fw_state[0], rnn.LSTMStateTuple):  # MultiLstmCell
	# 	encoder_state = tuple(map(
	# 		lambda fw_state, bw_state: rnn.LSTMStateTuple(
	# 			c=tf.concat((fw_state.c, bw_state.c), 1),
	# 			h=tf.concat((fw_state.h, bw_state.h), 1)),
	# 		encoder_fw_state, encoder_bw_state))
	# else:
	# 	encoder_state = tf.concat(
	# 		(encoder_fw_state, encoder_bw_state), 1)

	if USE_GPU :
		 cells = []
		 for i in range(decoding_layers):
			 cells.append(tf.contrib.rnn.DeviceWrapper(
					 tf.contrib.rnn.LSTMCell(2*num_hidden),
					 "/gpu:%d" % (2))) #(decoding_layers % num_gpus)))
	
		 decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
	else :    
		 decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(2*num_hidden) for _ in range(decoding_layers)])	
	
	
	projection_layer = tf.layers.Dense(vocab_size)  ## linear ---> Wx + b  
	attention_states = encoder_outputs
	#Size is [batch_size, max_time, num_units]

	#@TODO Please verify the correctness of memory sequence length
	# My worry is that since we are concatinaitng forward and backward, the length might not be accurate
	attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(2*num_hidden, 
		attention_states, memory_sequence_length=X_length)
	
	attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
		attention_mechanism, attention_layer_size=2*num_hidden)

	decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size)
	
	if is_train:   
		helper = tf.contrib.seq2seq.TrainingHelper(Y, Y_length)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			# initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(cell_state=encoder_state), output_layer=projection_layer)
			initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size), output_layer=projection_layer)
		output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
			maximum_iterations=max_out_time)
		# batch size x max out time x vocab size

	else: # Eval
		decoder = tf.contrib.seq2seq.BeamSearchDecoder(
		cell=decoder_cell,
		embedding = decoding_encoder,
		start_tokens= [start_token]*batch_size,
		end_token= end_token,
		initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
		beam_width=1,
		output_layer=projection_layer)

		output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
			maximum_iterations=max_out_time)
		#@TODO Length penalty weight gotta decide

	logits = output.rnn_output

	# decoder_inputs [max_decoder_time, batch_size]: target input words.
	# decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to 
	# the left by one time step with an end-of-sentence tag appended on the right.
	masks = tf.sequence_mask(Y_length, max_out_time, dtype=tf.float32, name='masks')
	loss_op = tf.contrib.seq2seq.sequence_loss(logits, y_shifted_input, masks)
	# crossent = tf.nn.sparse_softmax_cross_entropy_with_logits( \
	# 	labels=y_shifted_input, logits=logits)
	# target_weights = tf.sequence_mask(output_lengths, max_out_time, dtype=logits.dtype)
	# loss_op = (tf.reduce_sum(crossent * target_weights) /
	# 	tf.cast(batch_size, tf.float32))

	#Automatically updates variables
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)
	return x_input, y_input, y_shifted_input, X_length, Y_length, groups, logits, loss_op, optimizer

if __name__ == '__main__':

	# Implements RNN model for text normalization as defined in 
	# "RNN Approaches to Text Normalization : A challenge"
	# By Richard Sproat and Navdeep Jaitly
	# # Paper runs the model till they reach a perplexity of 1.0003.
	
	# Load configuration Parameters
	with open("config.yaml") as stream:
		params = yaml.load(stream)

	#@TODO https://github.com/tensorflow/tensorflow/issues/3420
	#Says more stacking is faster than bidirectional! We could try
	#Also can try GRU cell instead of LSTM 

	tf.reset_default_graph()

	cross_entropy_train = []
	perplexity_train = []
	cross_entropy_valid = []
	perplexity_valid = []

	best_loss = np.inf
	# Start training
	with tf.Session() as sess:

		print "Begin Training"
		x_input, y_input, y_shifted_input, X_length, Y_length, groups, logits, loss_op, optimizer, check = \
		BiRNN(params["num_hidden"], params["num_classes"], params["num_input"], params["num_groups"], \
			params["learning_rate"], params["encoding_layers"], params["vocab_size"], \
			params["decoding_layers"], params["start_token"], params["end_token"])

		
		# Save checkpoints on the way
		saver = tf.train.Saver()

		# Run the initializer
		initialize_all_variables()
		# print sess.run(tf.report_uninitialized_variables())

		for epoch in range(1, params["epochs"]+1):
			batch_cross_entropy_loss = 0.0
			global is_train
			is_train = True
			# saver.restore(sess, "results_model2/model_iter_100.cpkt")
			for step in range( 1 , params["num_train_batches"]+1):  
				data = np.load('data_model2/file_' + str(step) + '.npy').item()
				batch_x = data['batch_X']
				batch_y_shifted = data['batch_Y']
				x_length = data['X_length']
				y_length = data['Y_length']
				max_x = data['max_in']
				max_y = data['max_out']
				batch_groups = data['groups']

				# batch_x => batch_size x max_x x num_input
				# batch_y => batch_size x max_y
				# x_length => batch_size
				# y_length => batch_size
				# groups => batch_size

				batch_y = np.hstack((params["start_token"] * np.ones((batch_y_shifted.shape[0], 1)), batch_y_shifted[:, :-1]))
				feed_dict = {x_input:batch_x, X_length:x_length, y_input:batch_y, Y_length:y_length,
							 y_shifted_input:batch_y_shifted, groups:batch_groups}
				batch_loss, _ = sess.run([loss_op, optimizer], feed_dict)
				batch_cross_entropy_loss = batch_cross_entropy_loss + batch_loss


			batch_cross_entropy_loss = batch_cross_entropy_loss / params["num_train_batches"]
			print "Epoch %d Cross Entropy Error %f Perplexity Error %f" %(epoch,
				batch_cross_entropy_loss, math.exp(batch_cross_entropy_loss))

			if epoch % params["display_step"] == 0:
				filename = "results_model2/model_iter_"+str(epoch)+".cpkt"
				saver.save(sess, filename)
				cross_entropy_train.append(batch_cross_entropy_loss)
				perplexity_train.append(math.exp(batch_cross_entropy_loss))

				#Validation
				global is_train
				is_train = False

				valid_loss = 0.0
				for step in range(1, params["num_valid_batches"]+1):
					data = np.load('data_model2/valid_'+str(step)+'.npy').item() 
					valid_x = data['batch_X']
					valid_y_shifted = data['batch_Y']
					valid_x_length = data['X_length']
					valid_y_length = data['Y_length'] 
					max_x = data['max_in']
					max_y = data['max_out']
					batch_groups = data['groups']


					valid_y = np.hstack((params["start_token"] * np.ones((valid_y_shifted.shape[0], 1)), valid_y_shifted[:, :-1]))

					feed_dict = {x_input:valid_x, X_length:valid_x_length, y_input:valid_y, 
								Y_length:valid_y_length, y_shifted_input:valid_y_shifted, groups:batch_groups}
					predicted, valid_loss_step = sess.run([logits, loss_op], feed_dict)
					valid_loss = valid_loss + valid_loss_step

				valid_loss = valid_loss / params["num_valid_batches"]
				cross_entropy_valid.append(valid_loss)
				perplexity_valid.append(math.exp(valid_loss))

				if valid_loss < best_loss:
					best_loss = valid_loss
					saver.save(sess, 'results_model2/best_model.cpkt')

				print("Epoch " + str(epoch) + ", Train Loss= " + \
					  "{:.4f}".format(batch_cross_entropy_loss) + \
					  ", Validation Loss= " + "{:.4f}".format(valid_loss))

				np.save("results_model2/cross_entropy_train", cross_entropy_train)
				np.save("results_model2/cross_entropy_valid", cross_entropy_valid)
				np.save("results_model2/perplexity_train", perplexity_train)
				np.save("results_model2/perplexity_valid", perplexity_valid)

		print("Optimization Finished!")

		plt.figure(0)
		plt.plot(range(params["epochs"]//params["display_step"]), cross_entropy_valid, 'r', label="validation")
		plt.plot(range(params["epochs"]//params["display_step"]), cross_entropy_train, 'b', label="train")
		plt.legend()
		plt.ylabel('Cross Entropy Loss')
		plt.xlabel('Epochs')
		filename = "results_model2/cross_entropy_model1.png"
		plt.savefig(filename)
		plt.close()

		plt.figure(1)
		plt.plot(range(params["epochs"]//params["display_step"]), perplexity_valid, 'r', label="validation")
		plt.plot(range(params["epochs"]//params["display_step"]), perplexity_train, 'b', label="train")
		plt.legend()
		plt.ylabel('Perplexity')
		plt.xlabel('Epochs')
		filename = "results_model2/Perplexity_model1.png"
		plt.savefig(filename)
		plt.close()
