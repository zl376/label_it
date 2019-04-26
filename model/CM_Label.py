# Definition of
#   Labelling model using raw co-occurance matrix
#
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-21

from __future__ import absolute_import

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
import pickle

import utils

EPS = 1E-8



class CM_Label:
    '''
    
    '''
    def __init__(self, sparse=True):
        
        # DictVectorizer for embedding
        self.dv_x = DictVectorizer(sparse=sparse)
        self.dv_y = DictVectorizer(sparse=sparse)
        
        
    def build(self, universe_x, universe_y):
        # Build DictVectorizer for feature (x)
        self.dv_x.fit([ {x: 1} for x in universe_x ])
        self.map_v2i_x = self.dv_x.vocabulary_
        self.map_i2v_x = dict(zip(self.map_v2i_x.values(), self.map_v2i_x.keys()))        
        
        # Build DictVectorizer for target (y)
        self.dv_y.fit([ {x: 1} for x in universe_y ])
        self.map_v2i_y = self.dv_y.vocabulary_
        self.map_i2v_y = dict(zip(self.map_v2i_y.values(), self.map_v2i_y.keys()))
        
        
    def compile(self):
        # Deterministic, do nothing
        pass
        
        
    def fit(self, x=None,
                  y=None,
                  verbose=False):
        
        # Embed feature (x)
        embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
        
        # Embed target (y)
        embed_matrix_y = self.dv_y.transform([ {v: 1 for v in arr} for arr in y ])
        
        # Co-occurance matrix
        #   Raw
        co_matrix = embed_matrix_y.T.dot(embed_matrix_x)
        #   Normalized (column-wise)
        # co_matrix_norm = co_matrix / np.linalg.norm(co_matrix.A, ord=1, axis=0, keepdims=True)
        
        self.co_matrix = co_matrix
        
            
    def predict(self, x, n_best=1,
                         score=False):
        
        # Only support inference for SINGLE feature
        assert max([ len(arr) for arr in x ]) == 1, 'Only support inference for single feature'
            
        # Embed feature (x)
        embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
        
        # Match by finding NN in column of co-ocurrance matrix
        dist_matrix = self.co_matrix.dot(embed_matrix_x.T).A

        # y_idx = np.argmax(dist_matrix, axis=0)
        y_idx = np.argsort(dist_matrix, axis=0)[-n_best:, :].T
        y_score = np.sort(dist_matrix, axis=0)[-n_best:, :].T
        
        # Recover target (y) from embed idx
        y = utils.asarray_of_list([ [ self.map_i2v_y[i] for i in arr ] for arr in y_idx ])
        y_score = utils.asarray_of_list(y_score.tolist())
        
        if score:
            return y, y_score
        else:
            return y
        
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.co_matrix, file)
        
        
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.co_matrix = pickle.load(file)
