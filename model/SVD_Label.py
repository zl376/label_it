# Definition of
#   Labelling model using SVD
#
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-13

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import pickle

N_ITER = 10
RANDOM_STATE = 42



class SVD_Label:
    '''
    
    '''
    def __init__(self, n_components=20,
                       distance='cos',
                       sparse=True):
        self.n_components = n_components
        self.distance = distance
        
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
        embed_matrix_y = self.dv_y.transform([ {v: 1} for v in y ])
        
        # Co-occurance matrix
        #   Raw
        co_matrix = embed_matrix_y.T.dot(embed_matrix_x)
        #   Normalized (row-wise)
        co_matrix_norm = co_matrix / np.linalg.norm(co_matrix.A, ord=2, axis=1, keepdims=True)
        
        # Factorize using SVD
        svd = TruncatedSVD(n_components=self.n_components, n_iter=N_ITER, random_state=RANDOM_STATE)
        svd.fit(co_matrix)
        if verbose:
            print('Explained variance: {}'.format(np.sum(svd.explained_variance_ratio_)))
        
        self.U, self.Sigma, self.V = randomized_svd(co_matrix, 
                                                    n_components=self.n_components,
                                                    n_iter=N_ITER,
                                                    random_state=RANDOM_STATE)
        self.V = self.V.T
        
        if verbose:
            sns.lineplot(x=np.arange(len(self.Sigma))+1, y=self.Sigma)
            plt.title('Singular Value')
            
            
    def predict(self, x):
            
        # Embed feature (x)
        embed_matrix_x = self.dv_x.transform([ {v: 1 for v in arr} for arr in x ])
        
        # Transform embedded description into encoded space 
        enc_x = embed_matrix_x.dot(self.V)
        
        # Match by finding NN in encoded space wrt rows of U
        if self.distance == 'cos':
            # Cosine distance
            #   Normalize encoded vector 
            U_norm = self.U / np.linalg.norm(self.U, ord=2, axis=1, keepdims=True)
            enc_x_norm = enc_x / np.linalg.norm(enc_x, ord=2, axis=1, keepdims=True)
            
            dist_matrix = U_norm.dot(enc_x.T)

            y_idx = np.argmax(dist_matrix, axis=0)
        
        # Recover target (y) from embed idx
        y = np.asarray([ self.map_i2v_y[i] for i in y_idx ])
        
        return y
        
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.U, self.Sigma, self.V), file)
        
        
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.U, self.Sigma, self.V = pickle.load(file)