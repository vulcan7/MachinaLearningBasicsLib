import numpy as np
from sklearn.tree import DecisionTreeClassifier
import multiprocessing as mp
import random
#import workers
class rfc():
    def __init__(self,n_trees=100, max_depth=None,split_val_metric='Mean',
               min_info_gain=None,
               split_node_criterion='Gini', max_features='auto', bootstrap=False,
               n_cores=1):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.split_val_metric=split_val_metric
        self.min_info_gain=min_info_gain
        self.split_node_criterion=split_node_criterion
        self.bootstrap=bootstrap
        self.n_cores=n_cores
        self.max_features=max_features
        self.trees = []
    def fit(self,X,y):
        y=np.ravel(y)
        X=np.asarray(X)
        self.n_features_ = X.shape[1]
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        self.classes_ = []
        self.n_classes_ = []
        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        if self.max_features=='auto' or self.max_features=='sqrt':
            self.features=np.sqrt(self.n_features_)
        elif self.max_features=='log2':
            self.features=np.log2(self.n_features_)
        else:
            self.features=self.max_features
        self.features=int(self.features)
        self.forpredict = []
        proces = [None] * self.n_cores
        for i in range(self.n_trees):
            tree=DecisionTreeClassifier(
                                        )
            
            self.trees.append(tree)
            forpred = random.sample(range(self.n_features_), self.features)
            self.forpredict.append(forpred)
        pool = mp.Pool(processes=self.n_cores)
        results = [pool.apply(workers.cube, args=(self.trees[i],X[:,self.forpredict[i]],y,self.bootstrap)) for i in range(0,self.n_trees)]
        self.trees=results
    def predict(self,X):
        predictions = []
        X=np.array(X)
        #for i in range(self.n_trees):
         #   pred = self.trees[i].predict(X[:,self.forpredict[i]])
          #  predictions.append(pred)
        pool = mp.Pool(processes=self.n_cores)
        predictions =  [pool.apply(workers.pred, args=(self.trees[i],X[:,self.forpredict[i]])) for i in range(0,self.n_trees)]
        nclasses=self.n_classes_[0]
        nindex=X.shape[0]
        toreturn=[]
        predictions = np.array(predictions)
        for i in range(nindex):
            one = predictions[:,i]
            unique,counts = np.unique(one,return_counts=True)
            toreturn.append(unique[np.argmax(counts)])
        return toreturn

    def cube(tree,X,y,bootstrap):
        if bootstrap:
            n_samples= X.shape[0]
            indices = np.random.randint(0,n_samples,n_samples)
            X1 = []
            y1 =[]
            for i in indices:
                X1.append(X[i,:])
                y1.append(y[i,:])
            tree.fit(X1,y1)
        else:
            tree.fit(X,y)
        return tree
def pred(tree,X):
    return tree.predict(X)    
