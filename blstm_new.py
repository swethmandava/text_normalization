import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def BiRNN(num_hidden, num_classes, learning_rate, encoding_layers,
	decoding_layers, max_in_time, max_out_time, training=True):


	# Inputs
	#@TODO Gotta define encodings

	x_input = tf.placeholder(tf.float32, [max_in_time, batch_size])
	X = embedding_ops.embedding_lookup(embedding_encoder, x_input)
	X_length = tf.placeholder(tf.float32, [batch_size])
	y_input = tf.placeholder(tf.float32, [max_out_time, batch_size])
	Y = embedding_ops.embedding_lookup(decoding_encoder, y_input)
	Y_length = tf.placeholder(tf.float32, [batch_size])

# 
	# Reshape to match rnn.static_bidirectional_rnn function requirements
	# Current data input shape: (batch_size, max_in_time, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
	# X = tf.unstack(X, timesteps, 1)
	# Y = tf.unstack(Y, timesteps, 1)

	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	# Forward direction stacked lstm cell
	lstm_fw_cell = rnn_cell.MultiRNNCell([lstm_cell] * encoding_layers)
	# Backward direction stacked lstm cell
	lstm_bw_cell = rnn_cell.MultiRNNCell([lstm_cell] * encoding_layers)

	# Get lstm cell output
	try:
		encoder_outputs, _, _ = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X,
			X_length, dtype=tf.float32)
	except Exception: # Old TensorFlow version only returns outputs not states
		encoder_outputs = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X,
			X_length, dtype=tf.float32)

	# outputs is 2*[batch_size, max_time, num_units], one for forward and one for backward
	encoder_outputs = tf.concat(encoder_outputs[0], encoder_outputs[1], 2)
	# outputs is [batch_size, max_time, 2*num_units]


	decoder_cell = rnn_cell.MultiRNNCell([lstm_cell] * decoding_layers)
	projection_layer = layers_core.Dense(vocab_size)


	if training:
		attention_states = tf.transpose(encoder_outputs, [1,0,2])
		#Size is [batch_size, max_time, num_units]

		attention_mechanism = tf.contrib.seq2seq.LuongAtAttention(2*num_units, 
			attention_states, memory_sequence_length=X_length)

		helper = tf.contrib.seq2seq.TrainingHelper(Y, Y_length, time_major=True)

		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
			attention_mechanism, attention_layer_size=2*num_units)

		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			encoder_outputs, output_layer=projection_layer)

		output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_out_time)


		logits = output.rnn_output

		# @TODO Know what this Means? Y is decoder_inputs. Assuming Y_shifted as decoder_outputs
		# decoder_inputs [max_decoder_time, batch_size]: target input words.
		# decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to 
		# the left by one time step with an end-of-sentence tag appended on the right.
		
		# Cross entropy loss
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=Y_shifted, logits=logits)
		target_weights = tf.sequence_mask(output_lengths, max_out_time, dtype=logits.dtype)
		loss_op = (tf.reduce_sum(crossent * target_weights) /
			batch_size)

		#Automatically updates variables
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)
		return x_input, y_input, X_length, Y_length, crossent, loss_op, optimizer

	decoder = tf.contrib.seq2seq.BeamSearchDecoder(
	cell=decoder_cell,
	embedding=self.embedding_decoder,
	start_tokens=start_token,
	end_token=end_token,
	initial_state=encoder_outputs,
	beam_width=beam_width,
	output_layer=projection_layer,
	length_penalty_weight=length_penalty_weight)

	output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
		maximum_iterations=max_out_time)
	#@TODO Length penalty weight gotta decide
	# logits = tf.no_op()
	sample_id = outputs.predicted
	return x_input, X_length, sample_id
	



if __name__ == '__main__':

	# Paper runs the model till they reach a perplexity of 1.0003.
	# Stopping criterion can be that once it runs :P

	#@TODO
	# input_data = get_input() #How do we do this?
	# validation_data = get_input() 
	# test_data = get_input()

	learning_rate = 0.001 # Can do Cross Validation
	epochs = 100 # Need more?
	batch_size = 128 #@TODO Not sure if we can do batches
	display_step = 200

	num_input = 28 # Depending on character embeddings we get on google
	num_hidden = 128 # Given in paper
	encoding_layers = 4 #Giver in paper
	decoding_layers = 2 #Given in paper
	max_iter = 20 #@TODO can decide

	#@TODO https://github.com/tensorflow/tensorflow/issues/3420
	#Says more stacking is faster than bidirectional! We could try
	#Also can try GRU cell instead of LSTM
	num_classes = 256 + 2 # Number of possible characters, 256 ASCII
	# as well as start/end signals

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Save checkpoints on the way
	saver = tf.train.Saver()

	# Start training
	with tf.Session() as sess:
		x_input, y_input, X_length, Y_length, crossent, loss_op, optimizer = BiRNN(num_hidden, 
			num_classes, learning_rate, encoding_layers, decoding_layers, max_iter)
		# Run the initializer
		sess.run(init)

		for epoch in range(1, epochs+1):
			for batch_x, x_length, batch_y, y_length in input_data(): 
				#Define input_data of size 1 x timesteps x num_input : @TODO Can we do batches?
				# Define Y as timesteps_output x num_input. Include start and end tags for all
				# I "think" only end tags will also do if we don't have to learn embeddings.
				feed_dict = {x_input:batch_x, X_length:x_length, y_input:batch_y, Y_length:y_length}
				#Run and train
				batch_loss, _ = sess.run([loss_op, optimizer], feed_dict)
				if step % display_step == 0 or step == 1:
					print("Step " + str(step) + ", Minibatch Loss= " + \
						  "{:.4f}".format(loss))

		print("Optimization Finished!")


		#@TODO Validation and Test Should be quite similar