from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.tracking import tracking

import numpy as np
import tensorflow.compat.v2 as tf
import os
import tempfile
import re
import html 
import pickle

from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM,Input,Embedding
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D,concatenate

class LanguageModelEncoder(tf.train.Checkpoint):
    def __init__(self,vocab_size,emb_dim,state_size,n_layers):
        super(LanguageModelEncoder, self).__init__()
        self._state_size = state_size
        self.embedding_layer = Embedding(vocab_size,emb_dim)
        self._lstm_layers = [LSTM(self._state_size,return_sequences=True) for i in range(n_layers)]
        
    @tf.function(input_signature=[tf.TensorSpec([None,None], tf.dtypes.int64)])
    def __call__(self,sentence_lookup_ids):
        
        emb_output = self.embedding_layer(sentence_lookup_ids)
        lstm_output = emb_output # initialize to the input
        for lstm_layer in self._lstm_layers:
            lstm_output = lstm_layer(lstm_output)
        return lstm_output
        
        
        
def write_vocabulary_file(vocabulary):
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, "tokens.txt")
  with tf.io.gfile.GFile(vocabulary_file, "w") as f:
    for entry in vocabulary:
      f.write(entry + "\n")
  return vocabulary_file


class ULMFiTModule(tf.train.Checkpoint):
  """
  Trains a language model on given sentences
  """

  def __init__(self, vocab, emb_dim, buckets, state_size,n_layers):
    super(ULMFiTModule, self).__init__()
    self._buckets = buckets
    self._vocab_size = len(vocab)
    self.emb_row_size = self._vocab_size+self._buckets
    #self._embeddings = tf.Variable(tf.random.uniform(shape=[self.emb_row_size, emb_dim]))
    self._state_size = state_size
    self.model = LanguageModelEncoder(self.emb_row_size,emb_dim,state_size,n_layers)
    self._vocabulary_file = tracking.TrackableAsset(write_vocabulary_file(vocab)) 
    self.w2i_table = lookup_ops.index_table_from_file(
                    vocabulary_file= self._vocabulary_file,
                    num_oov_buckets=self._buckets,
                    hasher_spec=lookup_ops.FastHashSpec)
    self.i2w_table = lookup_ops.index_to_string_table_from_file(
                    vocabulary_file=self._vocabulary_file, 
                    delimiter = '\n',
                    default_value="UNKNOWN")
    self._logit_layer = tf.keras.layers.Dense(self.emb_row_size)
    self.optimizer = tf.keras.optimizers.Adam()


    
  def _tokenize(self, sentences):
    # Perform a minimalistic text preprocessing by removing punctuation and
    # splitting on spaces.
    normalized_sentences = tf.strings.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    sparse_tokens = tf.strings.split(normalized_sentences, " ").to_sparse()

    # Deal with a corner case: there is one empty sentence.
    sparse_tokens, _ = tf.sparse.fill_empty_rows(sparse_tokens, tf.constant(""))
    # Deal with a corner case: all sentences are empty.
    sparse_tokens = tf.sparse.reset_shape(sparse_tokens)

    return (sparse_tokens.indices, sparse_tokens.values,
            sparse_tokens.dense_shape)
    
  def _indices_to_words(self, indices):
    #return tf.gather(self._vocab_tensor, indices)
    return self.i2w_table.lookup(indices)
    

  def _words_to_indices(self, words):
    #return tf.strings.to_hash_bucket(words, self._buckets)
    return self.w2i_table.lookup(words)
  
  @tf.function(input_signature=[tf.TensorSpec([None],tf.dtypes.string)])   
  def _tokens_to_lookup_ids(self,sentences):
    token_ids, token_values, token_dense_shape = self._tokenize(sentences)
    tokens_sparse = tf.sparse.SparseTensor(
        indices=token_ids, values=token_values, dense_shape=token_dense_shape)
    tokens = tf.sparse.to_dense(tokens_sparse, default_value="")

    sparse_lookup_ids = tf.sparse.SparseTensor(
        indices=tokens_sparse.indices,
        values=self._words_to_indices(tokens_sparse.values),
        dense_shape=tokens_sparse.dense_shape)
    lookup_ids = tf.sparse.to_dense(sparse_lookup_ids, default_value=0)
    return tokens,lookup_ids
        
    

  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])
  def train(self, sentences):
    tokens,lookup_ids = self._tokens_to_lookup_ids(sentences)
    # Targets are the next word for each word of the sentence.
    tokens_ids_seq = lookup_ids[:, 0:-1]
    tokens_ids_target = lookup_ids[:, 1:]
    tokens_prefix = tokens[:, 0:-1]

    # Mask determining which positions we care about for a loss: all positions
    # that have a valid non-terminal token.
    mask = tf.logical_and(
        tf.logical_not(tf.equal(tokens_prefix, "")),
        tf.logical_not(tf.equal(tokens_prefix, "<E>")))

    input_mask = tf.cast(mask, tf.int32)

    with tf.GradientTape() as t:
      #sentence_embeddings = tf.nn.embedding_lookup(self._embeddings,tokens_ids_seq)
    
      lstm_output = self.model(tokens_ids_seq)
      lstm_output = tf.reshape(lstm_output, [-1,self._state_size])
      logits = self._logit_layer(lstm_output)
      

      targets = tf.reshape(tokens_ids_target, [-1])
      weights = tf.cast(tf.reshape(input_mask, [-1]), tf.float32)

      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits)

      # Final loss is the mean loss for all token losses.
      final_loss = tf.math.divide(
          tf.reduce_sum(tf.multiply(losses, weights)),
          tf.reduce_sum(weights),
          name="final_loss")

    watched = t.watched_variables()
    gradients = t.gradient(final_loss, watched)
    self.optimizer.apply_gradients(zip(gradients, watched))

    #for w, g in zip(watched, gradients):
    #  w.assign_sub(g)

    return final_loss
  
  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])  
  def validate(self,sentences):
    tokens,lookup_ids = self._tokens_to_lookup_ids(sentences)
    # Targets are the next word for each word of the sentence.
    tokens_ids_seq = lookup_ids[:, 0:-1]
    tokens_ids_target = lookup_ids[:, 1:]
    tokens_prefix = tokens[:, 0:-1]

    # Mask determining which positions we care about for a loss: all positions
    # that have a valid non-terminal token.
    mask = tf.logical_and(
        tf.logical_not(tf.equal(tokens_prefix, "")),
        tf.logical_not(tf.equal(tokens_prefix, "<E>")))

    input_mask = tf.cast(mask, tf.int32)

    lstm_output = self.model(tokens_ids_seq)
    lstm_output = tf.reshape(lstm_output, [-1,self._state_size])
    logits = self._logit_layer(lstm_output)
      

    targets = tf.reshape(tokens_ids_target, [-1])
    weights = tf.cast(tf.reshape(input_mask, [-1]), tf.float32)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits)

    # Final loss is the mean loss for all token losses.
    final_loss = tf.math.divide(
          tf.reduce_sum(tf.multiply(losses, weights)),
          tf.reduce_sum(weights),
          name="final_validation_loss")

    return final_loss
    
  @tf.function
  def decode_greedy(self, sequence_length, first_word):

    sequence = [first_word]
    current_word = first_word
    current_id = tf.expand_dims(self._words_to_indices(current_word), 0)

    for _ in range(sequence_length):
      lstm_output = self.model(tf.expand_dims(current_id,0))
      lstm_output = tf.reshape(lstm_output, [-1,self._state_size])
      logits = self._logit_layer(lstm_output)
      softmax = tf.nn.softmax(logits)

      next_ids = tf.math.argmax(softmax, axis=1)
      next_words = self._indices_to_words(next_ids)[0]
      
      current_id = next_ids
      current_word = next_words
      sequence.append(current_word)

    return sequence

