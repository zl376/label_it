# Definition of
#   Labelling model using (deep) Neural Network
#
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-18

from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import sparse_categorical_accuracy
from sklearn.feature_extraction import DictVectorizer
from gensim.models import TfidfModel
from datetime import datetime
from collections import deque
import os
import pickle

import label_it.utils as utils



class DNN_Label:
    '''
    TODO: Multi-GPU version
    '''
    def __init__(self, vocabulary_size, label_size,
                       param_layer=(20,),
                       sparse=True,
                       tfidf=False,
                       **kwargs):
        self.vocabulary_size = vocabulary_size
        self.label_size = label_size
        self.param_layer = param_layer
        self.tfidf = tfidf
        
        # DictVectorizer for embedding
        self.dv_x = DictVectorizer(sparse=sparse, sort=False)
        self.dv_y = DictVectorizer(sparse=sparse, sort=False)
        
        # Model for Tfidf
        self.TFIDF = None
        
        
    def build(self, universe_x, universe_y):
        # Build DictVectorizer for feature (x)
        self.dv_x.fit([ {x: 1} for x in universe_x ])
        self.map_v2i_x = self.dv_x.vocabulary_
        self.map_i2v_x = dict(zip(self.map_v2i_x.values(), self.map_v2i_x.keys()))        
        
        # Build DictVectorizer for target (y)
        self.dv_y.fit([ {x: 1} for x in universe_y ])
        self.map_v2i_y = self.dv_y.vocabulary_
        self.map_i2v_y = dict(zip(self.map_v2i_y.values(), self.map_v2i_y.keys()))
        
        # Reset graph node
        # tf.reset_default_graph()
        with self.graph.as_default():
            # Prepare input placeholder
            with tf.name_scope('inputs'):
                # self.x = tf.placeholder(tf.int32, shape=(None,), name='word')
                self.x = tf.sparse.placeholder(tf.int32, shape=(None, self.vocabulary_size), name='inputs')
                self.y = tf.placeholder(tf.int32, shape=(None,) + (1,), name='label')
            
            # Prepare embedding
            with tf.name_scope('embedding'):
                self.embeddings_weights = tf.Variable(tf.random_uniform([self.vocabulary_size, self.param_layer[0]], 
                                                                        -1.0, 1.0), name='word')
                self.embeddings_bias = tf.Variable(tf.zeros([self.param_layer[0]]))
                self.weights = tf.Variable(tf.truncated_normal([self.label_size, self.param_layer[-1]],
                                                               stddev=1.0 / np.sqrt(self.param_layer[-1])))
                self.biases = tf.Variable(tf.zeros([self.label_size]))
                                                                
            # Prepare intermediate layers
            with tf.name_scope('inter'):
                x = tf.nn.embedding_lookup_sparse(self.embeddings_weights, self.x, None, combiner='sum') \
                    + self.embeddings_bias
                x = tf.nn.relu(x)
                for n_feature in self.param_layer[1:]:
                    x = tf.layers.dense(x, n_feature, activation=tf.nn.relu)
            
            # Create embedded vector
            self.embed = x
            
            # Create logit
            self.logit = tf.matmul(self.embed, tf.transpose(self.weights)) + self.biases
            
            # Saver
            self.saver = tf.train.Saver(max_to_keep=10)
        
        
    def compile(self, num_sampled=5):
        
        with self.graph.as_default():
            # Construct loss
            with tf.name_scope('loss'):
                #   Use NCE loss for the batch.
                #   tf.nce_loss automatically draws a new sample of the negative labels each
                #   time we evaluate the loss.
                #   Explanation of the meaning of NCE loss:
                #       http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/                
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.weights,
                                                          biases=self.biases,
                                                          labels=self.y,
                                                          inputs=self.embed,
                                                          num_sampled=num_sampled,
                                                          num_classes=self.label_size))
            # Construct Metric
            with tf.name_scope('metric'):
                self.accuracy = tf.reduce_mean(sparse_categorical_accuracy(self.y, self.logit))
            
            # Construct optimizer
            with tf.name_scope('optimizer'):
                self.learning_rate = tf.Variable(1E-3, trainable=False, name="learning_rate")
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                
            # Summary
            self.loss_summary = tf.summary.scalar("loss/loss_train", self.loss)
            self.loss_val_summary = tf.summary.scalar("loss/loss_val", self.loss)
            self.accuracy_summary = tf.summary.scalar("metric/acc_train", self.accuracy)
            self.accuracy_val_summary = tf.summary.scalar("metric/acc_val", self.accuracy)
            
            # Saver
            self.saver = tf.train.Saver(max_to_keep=10)
            
            # Initialization
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
            
    def fit(self, x=None, 
                  y=None,
                  verbose=True):
        
        # Split into train/validation data
        validation_split = 0.1
        N = len(x)
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx_train, idx_val = idx[:-np.floor(N * validation_split).astype(np.int)], idx[-np.floor(N * validation_split).astype(np.int):]
        
        if self.tfidf:
            embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
            self.TFIDF = TfidfModel(list( [ (j, row[0,j]) for j in row.nonzero()[1] ] for row in embed_matrix_x ),
                                    normalize=False)
            x = np.asarray([ { self.map_i2v_x[i]: w
                               for i,w in self.TFIDF[list( (self.map_v2i_x[v], 1.0) 
                                                           for v in arr if v in self.map_v2i_x )] }
                             for arr in x ])
        else:
            x = np.asarray([ {v: 1 for v in arr} for arr in x ])
        
        # Construct generator
        def generator(batch_size):
            # Generate a positive batch
            idx = np.random.choice(idx_train, replace=False, size=(batch_size))
            return x[idx], y[idx]
        def generator_val(batch_size):
            # Use all validation data
            return x[idx_val], y[idx_val]
        
        # Fit using generator
        self.fit_generator(generator=generator, 
                           batch_size=32,
                           generator_val=generator_val,
                           epochs=1,
                           steps_per_epoch=100000,
                           logdir='./log',
                           ckptdir='./ckpt',
                           n_recent=1000,
                           verbose=verbose)
    
        
    def fit_generator(self, generator=None,
                            batch_size=None,
                            epochs=1,
                            verbose=True,
                            logdir=None,
                            ckptdir=None,
                            n_recent=100,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            generator_val=None,
                            batch_size_val=None,
                            freq_val=10,
                            **kwargs):
  
        # Initialize training
        if initial_epoch == 0:
            self.sess.run(self.init)
        if steps_per_epoch is None:
            steps_per_epoch = 1
        #   Progress
        freq_prog = 100
        progbar = Progbar(steps_per_epoch, stateful_metrics=['loss', 'val loss',
                                                             'accuracy', 'val accuracy'])
        #   Summary (Tensorboard)
        freq_log = 100
        if not logdir is None:
            logdir = os.path.join(logdir, datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))
            flag_log = True
            train_writer = tf.summary.FileWriter(logdir, self.graph, flush_secs=5)
        else:
            flag_log = False
        #   Checkpoint
        freq_ckpt = freq_val
        if not ckptdir is None:
            flag_ckpt = True
            fn_ckpt = os.path.join(ckptdir, 'WE_Label')
            # Save best model
        else:
            flag_ckpt = False
        def check_point(step):
            self.save_model(fn_ckpt, epoch=step)
        def load_best():
            self.load_model(tf.train.latest_checkpoint(ckptdir))
        #   History
        history = {'train': deque(maxlen=n_recent),
                   'val': deque(maxlen=n_recent)}
        def is_plateau():
            h = history['val'] if history['val'] else history['train']
            return len(h) == h.maxlen and h[0] == max(h)
        best = deque(maxlen=1)
        def check_best(loss):
            if not best or loss <= best[0]:
                best.append(loss)
                return True
            else:
                return False
        #   Validation
        if not generator_val is None:
            flag_val = True
            if batch_size_val is None:
                batch_size_val = batch_size
        else:
            flag_val = False
        #   Stopping
        min_lr = 1E-5
        
            
            
        # We must initialize all variables before we use them.
        self.sess.run(self.init)
        print('Initialized')

        for step in range(1, steps_per_epoch+1):
            def should(freq):
                return step == 1 or step%freq == 0
            
            batch_x, batch_y = self._preprocess_batch(*generator(batch_size))
            feed_dict = {self.x: batch_x, self.y: batch_y}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, loss, accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy],
                                              feed_dict=feed_dict)

            if flag_val and should(freq_val):
                batch_val_x, batch_val_y = self._preprocess_batch(*generator_val(batch_size_val))
                feed_dict = {self.x: batch_val_x, self.y: batch_val_y}                
                loss_val, accuracy_val = self.sess.run([self.loss, self.accuracy],
                                                       feed_dict=feed_dict)
                history['val'].append(-loss_val)
            else:
                history['train'].append(-loss)
            
            if verbose and should(freq_prog):
                progbar.update(step, [('loss', np.mean([loss])),
                                      ('val loss', np.mean([loss_val])),
                                      ('accuracy', np.mean([accuracy])),
                                      ('val accuracy', np.mean([accuracy_val])),])
                
            if flag_log and should(freq_log):
                for x in (self.sess.run(self.loss_summary, feed_dict={self.loss: loss}),
                          self.sess.run(self.accuracy_summary, feed_dict={self.accuracy: accuracy})):
                    train_writer.add_summary(x, step)
                if flag_val:
                    for x in (self.sess.run(self.loss_val_summary, feed_dict={self.loss: loss_val}),
                              self.sess.run(self.accuracy_val_summary, feed_dict={self.accuracy: accuracy_val})):
                        train_writer.add_summary(x, step)
            

            if flag_ckpt and should(freq_ckpt):
                val_check = loss_val if flag_val else loss
                if check_best(val_check):
                    print('\rSave BEST model at step {0} ({1:.3f})'.format(step, val_check), end='', flush=True)
                    check_point(step)
            
            if is_plateau():
                print('\nPlateau reached.'.format(step))
                # Load best model
                load_best()
                # Try reduce learning rate
                self.learning_rate = self.learning_rate * 0.5
                print('\nReduce lr to {}'.format(self.sess.run(self.learning_rate)))
                # Flush history
                [ x.clear() for x in history.values() ]
                if self.sess.run(self.learning_rate) < min_lr:
                    print('\nLearning rate {0} reaches limit: < {1}.'.format(self.sess.run(self.learning_rate), min_lr))
                    break
        
        # Make sure to load the best model
        load_best()

        # Get fit embedding
        self.Uw, self.Ub = self.sess.run([self.weights, self.biases], feed_dict={})
        #   Combine weight and bias (intercept) for U (label embedding)
        self.U = np.concatenate((self.Uw, self.Ub[..., np.newaxis]), axis=1)

    
    def predict(self, x, n_best=1):
        # Pre-embed feature (x)
        if self.tfidf:
            x = np.asarray([ { self.map_i2v_x[i]: w
                               for i,w in self.TFIDF[list( (self.map_v2i_x[v], 1.0) 
                                                           for v in arr if v in self.map_v2i_x )] }
                             for arr in x ])
        else:
            x = np.asarray([ {v: 1 for v in arr} for arr in x ])
        
        batch_x = self._preprocess_batch(x)
        
        # Directly inference to get logit output
        logit_matrix = self.sess.run(self.logit, feed_dict={self.x: batch_x})

        # y_idx = np.argmax(logit_matrix, axis=1)
        y_idx = np.argsort(logit_matrix, axis=1)[:, -n_best:]

        # Recover target (y) from embed idx
        y = utils.asarray_of_list([ [ self.map_i2v_y[i] for i in arr ] for arr in y_idx ])
        
        return y

    
    def save_model(self, filename, epoch=None):
        if not epoch is None:
            self.saver.save(self.sess, filename, global_step=epoch)
        else:
            self.saver.save(self.sess, filename, write_state=False)
    
        
    def load_model(self, filename, sess=None):
        sess = sess or self.sess
        try:
            self.saver.restore(sess, filename)
        except:
            self.saver.restore(sess)
        
    
    def _preprocess_batch(self, x, y=None):
        batch_size = x.shape[0]
        
        # Pre-embed feature (x)
        pre_embed_matrix_x = self.dv_x.transform(x.tolist())
        # Convert to SparseTensor
        coo = pre_embed_matrix_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        x_prep = tf.SparseTensorValue(indices, coo.col, coo.shape)   # Use coo.col for data, specified by         
                                                                     #   tf.nn.embedding_lookup_sparse

        # Pre-embed target (y)
        #   Just return the mapped index of y instead of embedded matrix
        if not y is None:
            y_prep = np.asarray([ [ self.map_v2i_y[v] for v in arr ] for arr in y ])
            return x_prep, y_prep
        else:
            return x_prep
    
    
    @property
    def graph(self):
        if not hasattr(self, '_graph'):
            self._graph = tf.Graph()
        return self._graph
    
    
    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess