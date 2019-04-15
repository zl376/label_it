# Definition of
#   Labelling model using Word Embedding (shallow Neural Network)
#
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-15

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
import os
import pickle



class WE_Label:
    '''
    TODO: Multi-GPU version
    '''
    def __init__(self, word_size, vocabulary_size, label_size,
                       embedding_size=20,
                       sparse=True,
                       dir_ckpt='./ckpt'):
        self.word_size = word_size
        self.vocabulary_size = vocabulary_size
        self.label_size = label_size
        self.embedding_size = embedding_size
        self.dir_ckpt = os.path.join(dir_ckpt)
        
        # DictVectorizer for embedding
        self.dv_x = DictVectorizer(sparse=sparse, sort=False)
        self.dv_y = DictVectorizer(sparse=sparse, sort=False)        
        
        if not os.path.exists(self.dir_ckpt):
            os.makedirs(self.dir_ckpt)
        
        
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
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='word')
                self.weights = tf.Variable(tf.truncated_normal([self.label_size, self.embedding_size], stddev=1.0 / np.sqrt(self.embedding_size)))
                self.biases = tf.Variable(tf.zeros([self.label_size]))
            
            # Create embedded vector
            self.embed = tf.nn.embedding_lookup_sparse(self.embeddings, self.x, None, combiner='sum')
            # self.embed = tf.matmul(self.x, self.embed_matrix_x)
        
        
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
            
            # Construct optimizer
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(self.loss)
            
            # Initialization
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
    
    def fit_generator(self, generator=None,
                            batch_size=None,
                            epochs=1,
                            verbose=True,
                            callbacks=None,
                            validation_split=0.,
                            validation_data=None,
                            shuffle=True,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,
                            freq_save=100,
                            **kwargs):
        
        if initial_epoch == 0:
            self.sess.run(self.init)
        
        if steps_per_epoch is None:
            steps_per_epoch = 1
        
        # We must initialize all variables before we use them.
        self.sess.run(self.init)
        print('Initialized')

        average_loss = 0
        for step in range(steps_per_epoch):
            def should(freq):
                return freq > 0 and step%freq == 0
            
            batch_x, batch_y = self._preprocess_batch(*generator(batch_size))
            feed_dict = {self.x: batch_x, self.y: batch_y}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, loss_val = self.sess.run([self.optimizer, self.loss],
                                        feed_dict=feed_dict,
                                        run_metadata=run_metadata)
            average_loss += loss_val

            if verbose and should(2000):
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000
                # batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        # Get fit embedding
        self.V, self.Uw, self.Ub = self.sess.run([self.embeddings, self.weights, self.biases], feed_dict={})
        #   Combine weight and bias (intercept) for U (label embedding)
        self.U = np.concatenate((self.Uw, self.Ub[..., np.newaxis]), axis=1)

    
    def predict(self, x):
        # Pre-embed feature (x)
        pre_embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
        
        # Transform embedded description into encoded space 
        enc_x = pre_embed_matrix_x.dot(self.V)
        #   Add intercept term for bias
        enc_x = np.concatenate((enc_x, np.ones(shape=(enc_x.shape[0], 1))), axis=1)
        
        # Match by finding maximized logit (log lik) wrt rows of U
        #   DO NOT Normalize encoded vector 
        logit_matrix = self.U.dot(enc_x.T)

        y_idx = np.argmax(logit_matrix, axis=0)

        # Recover target (y) from embed idx
        y = np.asarray([ self.map_i2v_y[i] for i in y_idx ])
        
        return y

    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.U, self.V), file)
        
        
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.U, self.V = pickle.load(file)
            
    
    def _preprocess_batch(self, x, y):
        batch_size = x.shape[0]
        
        # Pre-embed feature (x)
        pre_embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
        # Convert to SparseTensor
        coo = pre_embed_matrix_x.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        x_prep = tf.SparseTensorValue(indices, coo.col, coo.shape)   # Use coo.col for data, specified by         
                                                                     #   tf.nn.embedding_lookup_sparse

        # Pre-embed target (y)
        #   Just return the mapped index of y instead of embedded matrix
        y_prep = np.asarray([ self.map_v2i_y[v] for v in y ]).reshape(batch_size, 1)
        
        return x_prep, y_prep
    
    
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