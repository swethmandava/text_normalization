import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


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

def create_data() :
   
	#load X , Y , len_x , len_y
	#train_x =   
	#train_y =
	# len_x =
	#len_y =
	train_x = np.array([[1,2,3] , [4,5,6]])
	train_y = np.array([[1],[2]])
	len_x = np.array([3,3])
	len_y = np.array([1,1])
	#converting everything to tensors 
	# train_x = tf.convert_to_tensor(train_x, np.float32)
	# train_y = tf.convert_to_tensor(train_y, np.float32)
	# len_x = tf.convert_to_tensor(len_x, np.float32)
	# len_y = tf.convert_to_tensor(len_y, np.float32)
	#X = Dataset.from_tensor_slices((train_x, train_y)  
	return train_x , len_x , train_y , len_y


def process_data(train_x ,len_x ,train_y , len_y, batch_size):
	
	 train_x , len_x , train_y , len_y = tf.train.shuffle_batch(
	  [ train_x , len_x , train_y , len_y],
	  batch_size= batch_size,
	  num_threads=4,
	  capacity=50000,
	  min_after_dequeue=10000)
	 return train_x , len_x , train_y , len_y
 
	
def BiRNN(num_hidden, num_classes, learning_rate, encoding_layers, vocab_size,
	decoding_layers, max_in_time, max_out_time, beam_width, start_token,
	end_token):

	
	# Inputs
	#max_in_time --> encoder time steps
	#max_out_time --> decoder time steps
	#embedding_encoder --> embedding matrix
	
	embedding_encoder = tf.random_normal((num_classes,300 ))
	#decoding_encoder = tf.one_hot(vocab_size, vocab_size, dtype=tf.float32)
	decoding_encoder = tf.random_normal(( vocab_size, vocab_size))
	#@TODO Gotta define encodings
	#embedding_encoder = tf.get_variable("embeddings", shape=embedding_matrix.shape,  \
	#                          initializer=tf.constant_initializer(np.array(embedding_matrix)) , trainable=True)  
	
	x_input = tf.placeholder(tf.int32, [None, max_in_time])
	X = tf.nn.embedding_lookup(embedding_encoder, x_input)
	X_length = tf.placeholder(tf.int32, [None])
	y_input = tf.placeholder(tf.int32, [None, max_out_time])
	y_shifted_input = tf.placeholder(tf.int32, [None, max_out_time])
	Y = tf.nn.embedding_lookup(decoding_encoder, y_input) #### do we need this ###
	# start_token = tf.nn.embedding_lookup(decoding_encoder, start_token)
	# end_token = tf.nn.embedding_lookup(decoding_encoder, end_token)
	# print start_token

	# Y_shifted = tf.nn.embedding_lookup(decoding_encoder, y_shifted_input)
	# Y = y_input
	Y_length = tf.placeholder(tf.int32, [None])
	 
	# Reshape to match rnn.static_bidirectional_rnn function requirements
	# Current data input shape: (batch_size, max_in_time, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
	# X = tf.unstack(X, max_in_time, 1)
	# Y = tf.unstack(Y, timesteps, 1) 

	#lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
	batch_size = tf.shape(X)[0]
	num_gpus =3
	
	if USE_GPU :
		
		cells = []
		for i in range(encoding_layers):
			cells.append(tf.contrib.rnn.DeviceWrapper(
					tf.nn.rnn_cell.LSTMCell(num_hidden),
					"/gpu:%d" % (encoding_layers % num_gpus)))
		
		lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		
		
		lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		
	else :    

		# Forward direction stacked lstm cell
		lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_hidden) for _ in range(encoding_layers)])
		# Backward direction stacked lstm cell
		lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_hidden) for _ in range(encoding_layers)])

	((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state)) = \
	tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X, sequence_length=X_length, dtype=tf.float32)
	
	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 1)
	if isinstance(encoder_fw_state, rnn.LSTMStateTuple):  # LstmCell
		state_c = tf.concat(
			(encoder_fw_state.c, encoder_bw_state.c), 1)
		state_h = tf.concat(
			(encoder_fw_state.h, encoder_bw_state.h), 1)
		encoder_state = rnn.LSTMStateTuple(c=state_c, h=state_h)
	elif isinstance(encoder_fw_state, tuple) \
			and isinstance(encoder_fw_state[0], rnn.LSTMStateTuple):  # MultiLstmCell
		encoder_state = tuple(map(
			lambda fw_state, bw_state: rnn.LSTMStateTuple(
				c=tf.concat((fw_state.c, bw_state.c), 1),
				h=tf.concat((fw_state.h, bw_state.h), 1)),
			encoder_fw_state, encoder_bw_state))
	else:
		encoder_state = tf.concat(
			(encoder_fw_state, encoder_bw_state), 1)

	if USE_GPU :
		 cells = []
		 for i in range(decoding_layers):
			 cells.append(tf.contrib.rnn.DeviceWrapper(
					 tf.contrib.rnn.LSTMCell(2*num_hidden),
					 "/gpu:%d" % (decoding_layers % num_gpus)))
	
		 decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
	else :    
		 decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(2*num_hidden) for _ in range(decoding_layers)])
	
	print "ENCODER DONE"	
	
	
	projection_layer = tf.layers.Dense(vocab_size)  ## linear ---> Wx + b  
	attention_states = encoder_outputs
	#Size is [batch_size, max_time, num_units]

	attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(2*num_hidden, 
		attention_states) #, memory_sequence_length=X_length)

	# decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
	# 	attention_mechanism, attention_layer_size=2*num_hidden)
	
	attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
		attention_mechanism, attention_layer_size=2*num_hidden)
	decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size)
	
	if is_train:   
		print "DECODING"
		helper = tf.contrib.seq2seq.TrainingHelper(Y, Y_length)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			# initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(cell_state=encoder_state), output_layer=projection_layer)
			initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size), output_layer=projection_layer)
		output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_out_time)

	else: # Eval
		decoder = tf.contrib.seq2seq.BeamSearchDecoder(
		cell=decoder_cell,
		# embedding = decoding_encoder,
		start_tokens= [start_token]*batch_size,
		end_token= end_token,
		initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
		beam_width=1, #@TODO increase beam_width
		output_layer=projection_layer)

		output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
			maximum_iterations=max_out_time)
		#@TODO Length penalty weight gotta decide

	# sample_id = output.predicted
	logits = output.rnn_output

	# @TODO Know what this Means? Y is decoder_inputs. Assuming Y_shifted as decoder_outputs
	# decoder_inputs [max_decoder_time, batch_size]: target input words.
	# decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to 
	# the left by one time step with an end-of-sentence tag appended on the right.
	
	# Cross entropy loss
	# print y_shifted_input, logits
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits( \
		labels=y_shifted_input, logits=logits) ### i think it should be just Y 
	target_weights = tf.sequence_mask(output_lengths, max_out_time, dtype=logits.dtype)
	loss_op = (tf.reduce_sum(crossent * target_weights) /
		tf.cast(batch_size, tf.float32))
	#Automatically updates variables
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)
	return x_input, y_input, y_shifted_input, X_length, Y_length, logits, loss_op, optimizer
 

	

if __name__ == '__main__':

	# Paper runs the model till they reach a perplexity of 1.0003.
	# Stopping criterion can be that once it runs :P

	#@TODO
	# input_data = get_input() #How do we do this?
	# validation_data = get_input() 
	# test_data = get_input()
	learning_rate = 0.001 # Can do Cross Validation
	epochs = 100 # Need more?
	batch_size = 1 #128 #@TODO Not sure if we can do batches
	display_step = 200

	num_input = 300 # Depending on character embeddings we get on google ---> char embedding dimension
	num_hidden = 128 # Given in paper   ### wasn't it 256 ?
	encoding_layers = 4 #Given in paper
	decoding_layers = 2 #Given in paper
	max_iter = 20 #@TODO can decide
	vocab_size = 200 #number of decoder words we choose to keep in dicitonary
	max_in_time  = 3
	max_out_time = 1
	beam_width = 5 #@TODO check paper
	start_token = 0 #@TODO define
	end_token = 20 #@TODO define
	
	
	#@TODO https://github.com/tensorflow/tensorflow/issues/3420
	#Says more stacking is faster than bidirectional! We could try
	#Also can try GRU cell instead of LSTM
	num_classes =  95 + 2 #256 + 2 # Number of possible characters, 256 ASCII ---> if for 
	# as well as start/end signals

	tf.reset_default_graph()
	# Initialize the variables (i.e. assign their default value)
	# init = tf.global_variables_initializer()
	# init = tf.initialize_all_variables()

	# Save checkpoints on the way
	###saver = tf.train.Saver() @ERROR

	
	train_x , len_x , train_y , len_y = create_data()
	# train_x , len_x , train_y , len_y = process_data(train_x , len_x , train_y , len_y , batch_size)   
	# num_batches, _, _ = tf.shape(train_x)  
	num_batches = train_x.shape[0]//batch_size
	
	
	# Start training
	with tf.Session() as sess:
		global is_train
		is_train = True

		print "TRAINING"
		x_input, y_input, y_shifted_input, X_length, Y_length, logits, loss_op, optimizer = \
		BiRNN(num_hidden, num_classes, learning_rate, encoding_layers, vocab_size , \
			  decoding_layers, max_in_time, max_out_time, beam_width, start_token, end_token)
		# Run the initializer
		# sess.run(init)
		initialize_all_variables()
		print sess.run(tf.report_uninitialized_variables())
		print "KILL ME"
		# writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())
		print "INITIALIZED"
		for epoch in range(1, epochs+1):
			for step in range(num_batches): 

				batch_x = train_x[step * batch_size: (step+1)*batch_size, :]
				batch_y_shifted = train_y[step * batch_size: (step+1)*batch_size, :]
				x_length = len_x[step * batch_size: (step+1)*batch_size]
				y_length = len_y[step * batch_size: (step+1)*batch_size]

				#Define input_data of size 1 x timesteps x num_input : @TODO Can we do batches?
				# Define Y as timesteps_output x num_input. Include start and end tags for all
				# I "think" only end tags will also do if we don't have to learn embeddings.

				batch_y = np.hstack((start_token * np.ones((batch_y_shifted.shape[0], 1)), batch_y_shifted[:, :-1]))
				# print batch_y, batch_y_shifted
				# batch_y = np.copy(batch_y)
				feed_dict = {x_input:batch_x, X_length:x_length, y_input:batch_y, Y_length:y_length,
							 y_shifted_input:batch_y_shifted}
				#Run and train
				# add_summary = sess.run(writer)
				# writer.close()
				batch_loss, _ = sess.run([loss_op, optimizer], feed_dict)
				if step % display_step == 0 or step == 1:
					print("Step " + str(step) + ", Minibatch Loss= " + \
						  "{:.4f}".format(batch_loss))

		print("Optimization Finished!")

		#Validation

		global is_train
		is_train = False
		# valid_x, valid_x_length, valid_y, valid_y_length = validation_data
		valid_x, valid_x_length, valid_y_shifted, valid_y_length = create_data()
		valid_y = np.hstack((start_token * np.ones((valid_y_shifted.shape[0], 1)), valid_y_shifted[:, :-1]))

		feed_dict = {x_input:valid_x, X_length:valid_x_length, y_input:valid_y, 
					Y_length:y_length, y_shifted_input:valid_y_shifted}
		logits, loss_op = sess.run([logits, loss_op], feed_dict)

		print "Validation loss is:", loss_op
