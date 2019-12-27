"""
Created on wed Apr 17 00:01:45 2019
author:DAR 
"""
import numpy as np

class LogisticRegression(object):    
    """LogisticRegression 
    
    Parameters
    ------------
    learningRate : float, optional
        Constant by which updates are multiplied (falls between 0 and 1). Defaults to 0.01.
     
    numIterations : int, optional
        Number of passes (or epochs) over the training data. Defaults to 10.
    
    penalty : 'l1' or 'L2'
        Option to perform L2 or l1 regularization. Defaults to None.
    
    Attributes
    ------------
    weights : 1d-array, shape = [1, 1 + n_features]
        Weights after training phase
        
    iterationsPerformed : int
        Number of iterations of gradient descent performed prior to hitting tolerance level
        
    costs : list, length = numIterations
        Value of the log-likelihood cost function for each iteration of gradient descent
        
    References
    ------------
    https://en.wikipedia.org/wiki/Logistic_regression
    https://en.wikipedia.org/wiki/Regularization_(mathematics)
    
    """
    
    def __init__(self, learningRate, weights=None ,numIterations = 10, penalty = 'l2',lamda=1):
        
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.penalty = penalty
        self.weights=weights
        self.lamda = lamda
        
    def train(self, X_train, y_train):
        """Fit weights to the training data
        
        Parameters
        -----------
        X_train : {array-like}, shape = [n_samples, n_features]
            Training data to be fitted, where n_samples is the number of 
            samples and n_features is the number of features. 
            
        y_train : array-like, shape = [n_samples,], values = 1|0
            Labels (target values). 
        tol : float, optional
            Value indicating the weight change between epochs in which
            gradient descent should terminated. Defaults to 10 ** -4
        Returns:
        -----------
        self : object
        
        """
        if self.weights is None:
            self.weights = np.zeros(np.shape(X_train)[1] + 1)
        X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train]
        self.costs = []
        
        for i in range(self.numIterations):
            
            z = np.dot(X_train, self.weights)
            errors = y_train - logistic_func(z)
            if self.penalty is 'l1':            
                delta_w = self.learningRate * (self.lamda * np.dot(errors, X_train) + np.sum(np.sign(self.weights)))  
            elif self.penalty is 'l2':
                delta_w = self.learningRate * (self.lamda * np.dot(errors, X_train) + np.sum(self.weights))
                
            self.iterationsPerformed = i
            #>= tolerance
            if np.all(delta_w ): 
                #weight update
                self.weights += delta_w                                
                #Costs
                if self.penalty is not None:
                    self.costs.append(reg_logLiklihood(X_train, self.weights, y_train, self.lamda,self.penalty))
                else:
                    self.costs.append(logLiklihood(z, y_train))
            else:
                break
            
        return self
                    
    def predict(self, X_test):
        """predict class label 
        
        Parameters
        ------------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples
            and n_features is the number of features. n_features must
            be equal to the number of features in X_train.
        
        Returns
        ------------
        predictions : list, shape = [n_samples,], values = 1|0
            Class label predictions based on the weights fitted following 
            training phase.
        
        probs : list, shape = [n_samples,]
            Probability that the predicted class label is a member of the 
            positive class (falls between 0 and 1).
        
        """        
        z = self.weights[0] + np.dot(X_test, self.weights[1:])        
        probs = np.array([logistic_func(i) for i in z])
        predictions = np.where(probs >= 0.5, 1, 0)
       
        return predictions, probs
def logistic_func(z):   
    """Logistic (sigmoid) function, inverse of logit function
    
    Parameters:
    ------------
    z : float
        linear combinations of weights and sample features
        z = w_0 + w_1*x_1 + ... + w_n*x_n
    
    Returns:
    ---------
    Value of logistic function at z
    
    """
    return 1 / (1 + np.exp(-z))  
    
def logLiklihood(z, y):
    """Log-liklihood function (cost function to be minimized in logistic
    regression classification)
    
    Parameters
    -----------
    z : float
        linear combinations of weights and sample features
        z = w_0 + w_1*x_1 + ... + w_n*x_n
        
    y : list, values = 1|0
        target values
    
    Returns
    -----------
    Value of log-liklihood function with parameter z and target value y
    """
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z))))
    
    
def reg_logLiklihood(x, weights, y, lamda,penalty):
    """Regularizd log-liklihood function (cost function to minimized in logistic
    regression classification with L2 regularization)
    
    Parameters
    -----------
    x : {array-like}, shape = [n_samples, n_features + 1]
        feature vectors. Note, first column of x must be
        a vector of ones.
    
    weights : 1d-array, shape = [1, 1 + n_features]
        Coefficients that weight each samples feature vector
        
    y : list, shape = [n_samples,], values = 1|0
        target values
        
    C : float
        Regularization parameter.lambda    
    
    Returns
    -----------
    Value of regularized log-liklihood function with the given feature values,
    weights, target values, and regularization parameter
     
    """
    z = np.dot(x, weights) 
    if penalty=='l2':
        reg_term = (lamda/2) * np.dot(weights.T, weights)
    elif penalty=='l1':
        reg_term = (lamda/2) * np.linalg.norm(weights)

    
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z)))) + reg_term
