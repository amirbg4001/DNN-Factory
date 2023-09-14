from sklearn.model_selection import train_test_split
import pickle
import sys
import pandas as pd
import os
'''
This class is a Factory design pattern to produce several Machine Learning models and evaluate their performance for the model-informed approach.



'''
class MLmodels():
    def __init__(self):
        pass
    @classmethod
    def define(cls,model,options=None,loading=False):
        '''

        :param model : the type of the model is determined with this parameter. "svm" and "NN" are available models so far
        :param options : options parameter is a dictionary data type that defines the structure and configurations of the models
        :return: an instance of the defined model
        '''

        if model=="NN":
            from NN import NerualNet
            return NerualNet(options=options,loading=loading)
        if model=="svm":
            from svm import svm
            return svm(options=options,loading=loading)
        if model =="RandomForest":
            from RandomForest import RandomForest
            return RandomForest(options=options,loading=loading)
        if model =="RNN":
            from RecurrentNN import RecurrentNN
            return RecurrentNN(options=options,loading=loading)
    def data_preprocess(self,val_size,test_size,random_states,shuffle=False,custom_split=False,split_args = None):
        df = self.df
        X = df[self.features]
        y = df[self.target]
        if custom_split is None:
            X_train,X,y_train,y = train_test_split(X, y, test_size= val_size+test_size,random_state=random_states, shuffle=shuffle)
            X_val,X_test,y_val,y_test = train_test_split(X, y, test_size= test_size /(test_size+val_size), random_state=random_states, shuffle=shuffle)
        else:
            X_train , X_val,X_test , y_train , y_val , y_test = custom_split(X,y,val_size,test_size,split_args)
        if self.scaler is not None:
            if X_train.ndim == 1 :
                X_train = X_train.reshape((-1,1))
            if y_train.ndim == 1 :
                y_train = y_train.reshape((-1,1))
            self.input_scaler = self.scaler()
            self.target_scaler = self.scaler()
            X_train = self.input_scaler.fit_transform(X_train)
            X_test = self.input_scaler.transform(X_test)
            X_val = self.input_scaler.transform(X_val)
            y_train =self.target_scaler.fit_transform(y_train)
            y_test = self.target_scaler.transform(y_test)
            y_val = self.target_scaler.transform(y_val)
            if self.save_scaler:
                pickle.dump(self.input_scaler,open(os.path.join(self.save_scaler_address,self.save_scaler_name+"_input.pkl"),'wb'))
                pickle.dump(self.target_scaler,open(os.path.join(self.save_scaler_address,self.save_scaler_name+"_target.pkl"),'wb'))

        return {"X_train":X_train,"X_val":X_val,"X_test":X_test,"y_train":y_train,"y_val":y_val,"y_test":y_test}

    def predict(self, x_pred=None):
        '''

        :param x_pred: optional parameter if is not passed the function predicts based on the give dataset and prediction index
        :return:
        '''

        if x_pred is None:
            x_pred = self.df[self.features]
        if self.scaler is not None:
            x_pred = self.input_scaler.transform(x_pred)
            if len(self.features) ==1 : 
                return self.target_scaler.inverse_transform(self.model.predict(x_pred).reshape((-1,1)))
            else:
                return self.target_scaler.inverse_transform(self.model.predict(x_pred))
        else:
            if len(self.features) ==1 : 
                return self.model.predict(x_pred).reshape((-1,1))
            else:
                return self.model.predict(x_pred)

