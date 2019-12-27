import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from scipy import sparse
import scipy.sparse as sp

class NaiveBayesClassifier():
    def __init__(self,type1="Gaussian", priors=None):
        self.priors = priors
        self.type1=type1
    def predict(self, X):
        if(self.type1=="Gaussian"):
                joint_log_likelihood = []
                for i in range(np.size(self.classes_)):
                    jointi = np.log(self.class_prior_[i])
                    n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
                    n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                         (self.sigma_[i, :]), 1)
                    joint_log_likelihood.append(jointi + n_ij)
                joint_log_likelihood = np.array(joint_log_likelihood).T
                jll = joint_log_likelihood
                return self.classes_[np.argmax(jll, axis=1)]
        if(self.type1=='Multinominal'):
                jll = self.safe_sparse_dot(X, self.feature_log_prob_.T) +self.class_log_prior_
                return self.classes_[np.argmax(jll, axis=1)]
    def fit(self, X, y):
        if(self.type1=='Gaussian'):
            y=np.ravel(y)
            X=np.asarray(X, dtype=np.float64)
            self.classes_=np.unique(y)
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.sigma_ = np.zeros((n_classes, n_features))
            self.class_count_ = np.zeros(n_classes, dtype=np.float64)
            if self.priors is not None:
                priors = np.asarray(self.priors)
                    # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of'
                                         ' classes.')
                    # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                     raise ValueError('The sum of the priors should be 1.')
                    # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                    # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_),
                                                 dtype=np.float64)
            classes = self.classes_
            unique_y = np.unique(y)
            unique_y_in_classes = np.in1d(unique_y, classes)

            if not np.all(unique_y_in_classes):
                raise ValueError("The target label(s) %s in y do not exist in the "
                                 "initial classes %s" %
                                 (unique_y[~unique_y_in_classes], classes))

            for y_i in unique_y:
                i = classes.searchsorted(y_i)
                X_i = X[y == y_i, :]
                N_i = X_i.shape[0]
                new_theta, new_sigma = self._update_mean_variance(
                    self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
                    X_i)
                self.theta_[i, :] = new_theta
                self.sigma_[i, :] = new_sigma
                self.class_count_[i] += N_i
            if self.priors is None:
                # Empirical prior, with sample_weight taken into account
                self.class_prior_ = self.class_count_ / self.class_count_.sum()
            return self
        if(self.type1=='Multinominal'):
            _, n_features = X.shape
            
            Y = self.fit_transform1(y)
            self.classes_ = np.unique(y)
            if Y.shape[1] == 1:
                Y = np.concatenate((1 - Y, Y), axis=1)
            Y = Y.astype(np.float64)
            class_prior = self.priors
            n_effective_classes = Y.shape[1]
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)
            if np.any((X.data if issparse(X) else X) < 0):
                raise ValueError("Input X must be non-negative")
            self.feature_count_ += self.safe_sparse_dot(Y.T, X)
            self.class_count_ += Y.sum(axis=0)
            smoothed_fc = self.feature_count_ + 1.0
            smoothed_cc = smoothed_fc.sum(axis=1)
            self.feature_log_prob_ = (np.log(smoothed_fc) -
                                      np.log(smoothed_cc.reshape(-1, 1)))
            n_classes = len(self.classes_)
            if class_prior is not None:
                if len(class_prior) != n_classes:
                    raise ValueError("Number of priors must match number of"
                                     " classes.")
                self.class_log_prior_ = np.log(class_prior)

            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
            

    def _update_mean_variance(self,n_past, mu, var, X):
        if X.shape[0] == 0:
            return mu, var
        n_new = X.shape[0]
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)
        if n_past == 0:
            return new_mu, new_var
        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_past / float(n_new * n_total)) *
                     (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total
        return total_mu, total_var        
    def safe_sparse_dot(self,a, b):
        if sparse.issparse(a) or sparse.issparse(b):
            ret = a * b
            return ret
        else:
            return np.dot(a, b) 
    def fit_transform1(self,y):
        sparse_input_ = sp.issparse(y)
        classes= np.unique(y)
        n_samples = len(y)
        n_classes = len(classes)
        classes = np.asarray(classes)
        if n_classes == 1:
            Y = np.zeros((len(y), 1), dtype=np.int)
            return Y
        sorted_class = np.sort(classes)
        y = np.ravel(y)
        y_in_classes = np.in1d(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))
        data = np.empty_like(indices)
        data.fill(1)
        Y = sp.csr_matrix((data, indices, indptr),shape=(n_samples, n_classes))
        Y = Y.toarray()
        Y = Y.astype(int, copy=False)
        if np.any(classes != sorted_class):
            indices = np.searchsorted(sorted_class, classes)
            Y = Y[:, indices]
        return Y