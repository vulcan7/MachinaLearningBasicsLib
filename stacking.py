#from sklearn import tree
#from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
#import random
#from sklearn.metrics import accuracy_score

"""
data = pd.read_csv('/home/muralidhar/Iris.csv')
indices = data.index.tolist()
test_indices = random.sample(population=indices, k=50)
test_df = data.loc[test_indices]
train_df = data.drop(test_indices)

y_test = test_df[test_df.columns[-1]] #last column of data
y_train =train_df[train_df.columns[-1]]
x_train = train_df.drop(data.columns[-1],axis=1)
x_test = test_df.drop(data.columns[-1],axis=1)
x_tr = x_train
x_te = x_test
y_tr = y_train
y_te = y_test
"""
class Stacking():
    #global model_list
    def __init__(self,model_list):
        self.model_list = model_list
        #count = model_list
    #y_train = data[data.columns[-1]] #last column of data
    #x_train = data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)   
   # x_test = x_train
    def fit_predict(self,x_train,y_train,x_test):
        count = 0
        for model in (self.model_list):
            model_obj,count_new = model
        #count =count+count_new
        # stacked_dataset = list()
            for i in range(count_new):
                clf = model_obj
                clf = clf.fit(x_train,y_train)
           # for row in data:
             #   ans1 = clf.predict(row)
                y_pred = clf.predict(x_train) #training on total data set and x_train == x_test
           # ans.append(ans1)
                name_for_new_column = "column_num" + str(count) #giving a name for new column
                count +=1
                y_prd1 = clf.predict(x_test)
                x_train[name_for_new_column] = y_pred;  #appending new column to x_train
                x_test[name_for_new_column] = y_prd1;
    
        return x_test.iloc[:,-1] #last columnn of x_train gives the final prediction.
    
