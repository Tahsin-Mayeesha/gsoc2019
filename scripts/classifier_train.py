from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.tracking import tracking


from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
import os
import tempfile
import re
import html 

from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM,Input,Embedding
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D,concatenate
import pickle
from language_model import *

with open('data/imdb_reviews.pkl', 'rb') as f:
    imdb_reviews = pickle.load(f)
    
with open('data/imdb_labels.pkl', 'rb') as f:
    imdb_labels = pickle.load(f)
    

imdb_labels = tf.keras.utils.to_categorical(imdb_labels,2)

imdb_dataset = tf.data.Dataset.from_tensor_slices(imdb_reviews).batch(16,drop_remainder=True)
imdb_dataset_labels = tf.data.Dataset.from_tensor_slices(imdb_labels).batch(16,drop_remainder=True)
imdb_train_set = tf.data.Dataset.zip((imdb_dataset,imdb_dataset_labels))


sentences = ["<S> hello there <E>", "<S> how are you doing today <E>","<S> I am fine thank you <E>",
             "<S> hello world <E>", "<S> who are you? <E>"]
validation_sentences = ["<S> hello there <E>", "<S> how are you doing today <E>","<S> I am fine thank you <E>"]
vocab = [
      "<S>", "<E>", "hello", "there", "how", "are", "you", "doing", "today","I","am","fine","thank","world","who"]

module = ULMFiTModule(vocab=vocab, emb_dim=10, buckets=1, state_size=128,n_layers=1)

# bug : module needs to be called at least once before the savedmodel loaded gradients gets updated later
for epoch in range(1):
    train_loss = module.train(tf.constant(sentences))
    validation_loss = module.validate(tf.constant(validation_sentences))
    print("Epoch ",epoch," Train loss: ",train_loss.numpy()," Validation loss ",validation_loss.numpy())
    
    
module = tf.saved_model.load("saved_models/finetuned_language_model")


class LanguageClassifier(Model):
    def __init__(self,language_module,num_labels,dense_units=(128,128),dropouts=(0.1,0.1)):
        
        # initialization stuff
        super(LanguageClassifier,self).__init__()
        self._language_module = language_module
        self.model_encoder = language_module.model
        
        
        # classifier head layers
        self.dense_layers = [Dense(units,activation="relu") for units in dense_units]
        self.dropout_layers = [Dropout(p) for p in dropouts]
        self.max_pool_layer = GlobalMaxPooling1D()
        self.average_pool_layer = GlobalAveragePooling1D()
        self.batchnorm_layer = BatchNormalization()
        self.n_layers = len(self.dense_layers)
        self.final_layer = Dense(num_labels,activation="sigmoid")
        
    def __call__(self,sentences):
        
        tokens,lookup_ids = self._language_module._tokens_to_lookup_ids(sentences)
        self.enc_out = self.model_encoder(lookup_ids)
        last_h = self.enc_out[:,-1,:]
        max_pool_output = self.max_pool_layer(self.enc_out)
        average_pool_output = self.average_pool_layer(self.enc_out)
        
        output = concatenate([last_h,max_pool_output,average_pool_output])
        
        for i in range(self.n_layers):
            output = self.dense_layers[i](output)
            #output = self.dropout_layers[i](output)
            output = self.batchnorm_layer(output)
        
        final_output = self.final_layer(output)
        return final_output        
    
model = LanguageClassifier(language_module=module,num_labels=2)

loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

train_loss_hist = []
train_accuracy_hist = []

def track(tl_score,tl_accuracy):
    train_loss_hist.append(tl_score)
    train_accuracy_hist.append(tl_accuracy)

@tf.function
def train_step(samples, labels):
  with tf.GradientTape() as tape:
    predictions = model(samples)
    loss = loss_object(labels, predictions)
  watched = tape.watched_variables()
  gradients = tape.gradient(loss, watched)
  optimizer.apply_gradients(zip(gradients, watched))

  train_loss(loss)
  train_accuracy(labels, predictions)
  return loss, train_accuracy(labels,predictions)
    
    
@tf.function
def test_step(samples, labels):
  predictions = model(samples)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

    
EPOCHS = 2
step = 0
for epoch in range(EPOCHS):
    for reviews,labels in imdb_train_set:
        loss,acc = train_step(reviews, labels)
        track(loss,acc)
        if step%500==0:
            print("Step ",step, " loss ",loss," Accuracy ", acc.numpy()*100)
        step+=1
        
model.save("classifer")

