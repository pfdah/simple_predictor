from abc import ABC, abstractmethod
import numpy as np
from numpy import log,dot,e,shape
import pandas as pd
import pickle


class BaseClass(ABC):
    def fit(self):
        pass

    def predict(self):
        pass

    def loss(self):
        pass


class MultiLogistic(BaseClass):
    def __init__(self, iteration = 300, learning_rate = 0.001, threshold = 0.0099):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.thres = threshold
        
    
    def fit(self,X,y,verbose=False):
        '''Fits X and y into logistic regression

        Parameters
        ----------
        X : ndarray 
            array of feature variables
        y : array
            array of target variable, that has been encoded
        verbose : bool, optional
            Display iterations and epochs, by default False
        
        Returns
        -------
        self: object 
            Returns object that has been trained for logistic regression
        '''
        self.verbose = verbose

        # Encode y
        y = np.array(pd.get_dummies(y))
        X = np.array(X)
        
        # Adding Bias Column
        X = np.insert(X,0,1,axis=1) 

        # Creating weights matrix
        self.weights = np.zeros(shape=(X.shape[1],y.shape[1]))
        
        # Fitting process
        for i in range(self.iteration):
            z = np.dot(X, self.weights) 
            h = self.softmax_activate(z) 
            self.weights -= self.learning_rate * np.dot(X.T, (h - y))/y.size
            
            if i % 1000 == 0 and verbose: 
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
        return self

    def predict(self,X):
        pred_prob = self.predict_prob(X)
        pred_argmax = np.argmax(pred_prob, axis = 1)
        return pred_argmax

    def predict_prob(self, X):
        X = np.insert(X,0,1,axis=1) 
        return self.softmax_activate(np.dot(X, self.weights))
        
    def softmax_activate(self,Z):
        return np.array([np.exp(z)/np.sum(np.exp(z)) for z in Z])

    def save_model(self,model_name='model.pkl'):
        '''Save the model into a pickle file

        Parameters
        ----------
        model_name : str, optional
            the name of the pickle file that we are trying to save, by default 'model.pkl'
        '''
        with open(model_name, 'wb') as handle:
            pickle.dump(self, handle)




def load_model(model_name='model.pkl'):
    '''Load the model from a pickle file

    Parameters
    ----------
    model_name : str, optional
        the name of the pickle file that we are trying to load, by default 'model.pkl'

    Returns
    -------
    object
        the model loaded into an object
    '''
    try:
        return pickle.load(open(model_name,'rb'))
    except:
        print("Cannot find the specified file ",model_name)
