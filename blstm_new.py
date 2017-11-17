import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


path = ''
data_exists = 1

   
def create_data() :
   
    #load X , Y , len_x , len_y
    #train_x =   
    #train_y =
    # len_x =
    #len_y =
    train_x = np.array([[1,2,3] , [4,5,6]])
    train_y = np.array([1,2])
    len_x = np.array([3,3])
    len_y = np.array([1,1])
    #converting everything to tensors 
    train_x = tf.convert_to_tensor(train_x, np.float32)
    train_y = tf.convert_to_tensor(train_y, np.float32)
    len_x = tf.convert_to_tensor(len_x, np.float32)
    len_y = tf.convert_to_tensor(len_y, np.float32)
    #X = Dataset.from_tensor_slices((train_x, train_y)  
    return train_x , len_x , train_y , len_y


def process_data(train_x , len_x , train_y , len_y , batch_size)   :
    
     train_data = tf.train.shuffle_batch(
      [ train_x , len_x , train_y , len_y],
      batch_size= batch_size,
      num_threads=4,
      capacity=50000,
      min_after_dequeue=10000)
     return train_data
 
    
def BiRNN(num_hidden, num_classes, learning_rate, encoding_layers, vocab_size , 
    decoding_layers, max_in_time, max_out_time, training=True):

    
    # Inputs
    #max_in_time --> encoder time steps
    #max_out_time --> decoder time steps
    #embedding_encoder --> embedding matrix
    
    embedding_matrix = np.zeros((num_classes,300 ))
    decoding_encoder = np.zeros(( vocab_size, 1))
    #@TODO Gotta define encodings
    embedding_encoder = tf.get_variable("embeddings", shape=embedding_matrix.shape,  \
                                 initializer=tf.constant_initializer(np.array(embedding_matrix)) , trainable=False)  
    
    x_input = tf.placeholder(tf.int32, [ batch_size , max_in_time  ])
    X = tf.nn.embedding_lookup(embedding_encoder, x_input)
    X_length = tf.placeholder(tf.int32, [batch_size])
    y_input = tf.placeholder(tf.int32, [max_out_time, batch_size])
    Y = tf.nn.embedding_lookup(decoding_encoder, y_input) #### do we need this ###
    Y_length = tf.placeholder(tf.int32, [batch_size])
    #length_penalty_weight =
    beam_width = 3
    start_token = 'SOS' 
    end_token = 'EOS'
     
    # Reshape to match rnn.static_bidirectional_rnn function requirements
    # Current data input shape: (batch_size, max_in_time, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
    # X = tf.unstack(X, timesteps, 1)
    # Y = tf.unstack(Y, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Forward direction stacked lstm cell
    lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * encoding_layers)
    # Backward direction stacked lstm cell
    lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * encoding_layers)

    # Get lstm cell output
    # X --->  batch ,time_step , embed_dim
    try:
        encoder_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X,
            X_length, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        encoder_outputs , hs = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X,
            X_length, dtype=tf.float32)

    # outputs is 2*[batch_size, max_time, num_units], one for forward and one for backward
    encoder_outputs = tf.concat(encoder_outputs,2)
    # outputs is [batch_size, max_time, 2*num_units]


    decoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * decoding_layers)
    projection_layer = tf.layers.Dense(vocab_size)  ## linear ---> Wx + b 
    ## @ERROR - layers_core not found changed it to layers
 
    attention_states = tf.transpose(encoder_outputs, [0,1,2])
    #Size is [batch_size, max_time, num_units]

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(2*num_hidden, 
        attention_states, memory_sequence_length=Y_length) ### changed to Y_length -- dont know if it makes sense but it works

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
        attention_mechanism, attention_layer_size=2*num_hidden)
        
    if training: 


        helper = tf.contrib.seq2seq.TrainingHelper(Y, Y_length, time_major=True)


        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
            encoder_outputs, output_layer=projection_layer)

        output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_out_time)


        logits = output.rnn_output

        # @TODO Know what this Means? Y is decoder_inputs. Assuming Y_shifted as decoder_outputs
        # decoder_inputs [max_decoder_time, batch_size]: target input words.
        # decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to 
        # the left by one time step with an end-of-sentence tag appended on the right.
        
        # Cross entropy loss
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            labels=Y_shifted, logits=logits) ### i think it should be just Y 
        target_weights = tf.sequence_mask(output_lengths, max_out_time, dtype=logits.dtype)
        loss_op = (tf.reduce_sum(crossent * target_weights) /
            batch_size)

        #Automatically updates variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)
        return x_input, y_input, X_length, Y_length, crossent, loss_op, optimizer

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    cell=decoder_cell,
    #embedding = embedding_decoder, -- dont think this is reqquired , can uncomment later
    start_tokens= start_token,
    end_token= end_token,
    initial_state=encoder_outputs,
    beam_width=beam_width,
    output_layer=projection_layer,
    length_penalty_weight=length_penalty_weight)

    output, _, output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
        maximum_iterations=max_out_time)
    #@TODO Length penalty weight gotta decide
    # logits = tf.no_op()
    sample_id = output.predicted
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
    batch_size = 1 #128 #@TODO Not sure if we can do batches
    display_step = 200

    num_input = 28 # Depending on character embeddings we get on google
    num_hidden = 128 # Given in paper   ### wasn't it 256 ?
    encoding_layers = 1 #Giver in paper
    decoding_layers = 2 #Given in paper
    max_iter = 20 #@TODO can decide
    vocab_size = 200 
    max_in_time  = 3
    max_out_time = 1
    
    
    #@TODO https://github.com/tensorflow/tensorflow/issues/3420
    #Says more stacking is faster than bidirectional! We could try
    #Also can try GRU cell instead of LSTM
    num_classes =  95 + 2 #256 + 2 # Number of possible characters, 256 ASCII ---> if for 
    # as well as start/end signals

    tf.reset_default_graph()
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Save checkpoints on the way
    ###saver = tf.train.Saver() @ERROR

    
    train_x , len_x , train_y , len_y = create_data()
    train_data = process_data(train_x , len_x , train_y , len_y , batch_size)     
    
    
    # Start training
    with tf.Session() as sess:
        x_input, y_input, X_length, Y_length, crossent, loss_op, optimizer = \
        BiRNN(num_hidden, num_classes, learning_rate, encoding_layers, vocab_size , \
              decoding_layers, max_in_time, max_out_time)
        # Run the initializer
        sess.run(init)

        for epoch in range(1, epochs+1):
            for step , batch_x, x_length, batch_y, y_length in enumerate(train_data): 
                #Define input_data of size 1 x timesteps x num_input : @TODO Can we do batches?
                # Define Y as timesteps_output x num_input. Include start and end tags for all
                # I "think" only end tags will also do if we don't have to learn embeddings.
                feed_dict = {x_input:batch_x, X_length:x_length, y_input:batch_y, Y_length:y_length}
                #Run and train
                batch_loss, _ = sess.run([loss_op, optimizer], feed_dict)
                if step % display_step == 0 or step == 1:
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(batch_loss))

        print("Optimization Finished!")


        #@TODO Validation and Test Should be quite similar