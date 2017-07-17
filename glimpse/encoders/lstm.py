import datetime
import tensorflow as tf
from helpers.definitions import data_dir
from glimpse.neural_net import NeuralNet
from lstm_helpers.projection_op import ProjectionOp


class LSTM(NeuralNet):
  
  def __init__(self):
    NeuralNet.__init__(self)

    # self.text_data = text_dataset  # Keep a reference on the dataset
    self.dtype = tf.float32

    # Placeholders
    self.encoder_inputs = None
    self.decoder_inputs = None  # Same that decoderTarget plus the <go>
    self.decoder_targets = None
    self.decoder_weights = None  # Adjust the learning to the target sentence size
    
    # Main operators
    self.loss_fct = None
    self.opt_op = None
    self.outputs = None  # Outputs of the network, list of probability for each words

    # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
    self.output_projection = None

    # Tensorflow utilities for convenience saving/logging
    self.writer = None
    self.saver = None
    self.global_step = 0  # num iterations for current model

    # Tensorflow main session (keep track for the daemon)
    self.sess = None

    self.build_nn()
    
  def build_nn(self):
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if 0 < self.params.softmax_samples < self.text_data.get_vocabulary_size():
      self.output_projection = ProjectionOp((self.params.hidden_size, self.text_data.get_vocabulary_size()),
                                            scope='softmax_projection',
                                            dtype=self.dtype)
      
    enco_deco_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params.hidden_size, state_is_tuple=True)
    enco_deco_cell = tf.nn.rnn_cell.MultiRNNCell([enco_deco_cell] * self.params.num_layers, state_is_tuple=True)
    
    mle = self.params.max_length_enco  # Batch size * sequence length * input dim
    mld = self.params.max_length_deco  # Same sentence length for input and output
    
    # Network input (placeholders)
    with tf.name_scope('placeholder_encoder'):
      self.encoder_inputs = [tf.placeholder(tf.int32, [None, ]) for _ in range(mle)]
    
    with tf.name_scope('placeholder_decoder'):
      self.decoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(mld)]
      self.decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(mld)]
      self.decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(mld)]
    
    # Define the network.
    # Here we use an embedding model, it takes integer as input and converts them into word vector for
    # better word representation.
    decoder_outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
      self.encoder_inputs,  # List<[batch=?, inputDim=1]>, list of size params.maxLength
      self.decoder_inputs,  # For training, we force the correct output (feed_previous=False)
      enco_deco_cell,
      self.text_data.get_vocabulary_size(),
      self.text_data.get_vocabulary_size(),  # Both encoder and decoder have the same number of class
      embedding_size=self.params.embedding_size,  # Dimension of each word
      output_projection=self.output_projection if self.output_projection else None,
      feed_previous=bool(self.params.test)
      # When we test (self.params.test), we use previous output as next input (feed_previous)
    )
    
    # For testing only
    if self.params.test:
      if not self.output_projection:
        self.outputs = decoder_outputs
      else:
        self.outputs = [self.output_projection(output) for output in decoder_outputs]
 
    # For training only
    else:
      # Finally, we define the loss function
      self.loss_fct = tf.nn.seq2seq.sequence_loss(
        decoder_outputs,
        self.decoder_targets,
        self.decoder_weights,
        self.text_data.get_vocabulary_size(),
        softmax_loss_function=self.sampled_softmax if self.output_projection else None  # If None, use default SoftMax
      )
      
      # Keep track of the cost
      tf.scalar_summary('loss', self.loss_fct)
      
      # Initialize the optimizer
      opt = tf.train.AdamOptimizer(
        learning_rate=self.params.lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08
      )
      
      self.opt_op = opt.minimize(self.loss_fct)

  def sampled_softmax(self, inputs, labels):
    labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_wt = tf.cast(tf.transpose(self.output_projection.W), tf.float32)
    local_b = tf.cast(self.output_projection.b, tf.float32)
    local_inputs = tf.cast(inputs, tf.float32)

    return tf.cast(
      tf.nn.sampled_softmax_loss(
        local_wt,  # Should have shape [num_classes, dim]
        local_b,
        local_inputs,
        labels,
        self.params.softmax_samples,  # The number of classes to randomly sample per batch
        self.text_data.get_vocabulary_size()  # The number of classes
      ),
      self.dtype
    )

  # Forward/training step operation.
  # Does not perform run on itself but just returns the operators to do so. Those then have to be run.
  # Args:
  # 	batch (Batch): Input data on testing mode, input and target on output mode
  # Return:
  # 	(ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dict
  def step(self, batch):
    # Feed the dictionary
    feed_dict = {}
    ops = None
    
    # Training
    if not self.params.test:
      for i in range(self.params.max_length_enco):
        feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]
      
      for i in range(self.params.max_length_deco):
        feed_dict[self.decoder_inputs[i]] = batch.decoder_seqs[i]
        feed_dict[self.decoder_targets[i]] = batch.target_seqs[i]
        feed_dict[self.decoder_weights[i]] = batch.weights[i]
      
      ops = (self.opt_op, self.loss_fct)
    
    # Testing (batchSize == 1)
    else:
      for i in range(self.params.max_length_enco):
        feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]
      
      feed_dict[self.decoder_inputs[0]] = [self.text_data.go_token]
      
      ops = (self.outputs,)
    
    # Return one pass operator
    return ops, feed_dict

  def train(self):
    # self.text_data = TextData(self.params)
  
    # Saver/summaries
    self.writer = tf.train.SummaryWriter(data_dir)
    self.saver = tf.train.Saver(max_to_keep=200)  # Arbitrary limit ?
  
    # Running session
    self.sess = tf.Session()
  
    print('Initializing variables...')
    
    self.sess.run(tf.initialize_all_variables())

    # Limit the number of training samples
    self.text_data.make_lighter(self.params.ratio_dataset)

    # Define the summary operator (Warning: Won't appear on the tensorboard graph)
    merged_summaries = tf.merge_all_summaries()

    if self.global_step == 0:  # Not restoring from previous run
      self.writer.add_graph(self.sess.graph)  # First time only

    # If restoring a model, restore the progression bar and the current batch?

    print('Start training (press Ctrl+C to save and exit)...')

    try:
      # Train!
      for e in range(self.params.num_epochs):
        print("\n----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.params.num_epochs, self.params.lr))
    
        batches = self.text_data.get_batches()
    
        tic = datetime.datetime.now()
        for next_batch in batches:
          # Training pass
          ops, feed_dict = self.step(next_batch)
      
          # training, loss
          assert len(ops) == 2
      
          _, loss, summary = self.sess.run(ops + (merged_summaries,), feed_dict)
      
          self.writer.add_summary(summary, self.global_step)
          self.global_step += 1
      
          # Reached a checkpoint
          if self.global_step % self.params.save_every == 0:
            self.save_session(self.sess)
    
        toc = datetime.datetime.now()
    
        # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
        print("Epoch finished in {}".format(toc - tic))

    # Exit program if user presses Ctrl+C while testing
    except (KeyboardInterrupt, SystemExit):
      print('Interruption detected, exiting the program...')

    # Ultimate saving before complete exit
    self.save_session(self.sess)
    self.sess.close()
    
  def save_session(self, sess):
    print 'Checkpoint reached: saving model (don\'t stop the run)...'
    self.saver.save(sess, '{}/lstm_enc.ckpt'.format(data_dir))
    print 'Model saved.'

  # On which device should we run the model?
  def get_device(self):
    device = self.params.device

    valid_devices = {
      'cpu': '/cpu:0',
      'gpu': '/gpu:0'
    }

    if not device or device not in valid_devices.keys():
      return None

    return valid_devices[device]