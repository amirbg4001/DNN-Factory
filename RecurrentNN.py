from keras.models import Sequential,load_model
import sys
import os
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from MLModel import MLmodels
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy  as np
import json
'''
    RNN class is a Decorator pattern for RNN architecture implemmented by Tensorflow.
    Decorator design pattern solves the need of specific behaviour of an object.In this project the immplemented network is used as a the AI part
    of the model-informed model
    
    This model is configured for Time-series predictions ,
    TODO Encoder-Decoder 
    INIT : 

       1-  The "Options" parameter has the structure bellow , this parameter is the mandatory configuration of these class.

        Options = {
            "df" : Pandas based dataframe of the whole dataset --> Compulsory <dtype : Pandas.DataFrame>
            "features" : data labels for the RNN (X) --> Compulsory <dtype : list of strings> , ["Independant Feature 1","Independant Feature 2",..""Independant Feature n", "Target dependent Feature 1","Target dependent Feature 2",...,"Target dependent Feature n"]
            "targets" : the targets for RNN (Y) --> Compulsory <dtype : list of strings> ,
            "length" : Indicates the memory length of the RNN --> Compulsory <dtype : Int> ,
            "horizon: Indicates the prediction horizon of the RNN --> Compulsory <dtype : Int> ,
            "layers" : an ordered list of the Neural Network Layers and their activation function   --> Compulsory <dtype : List> ,
            "optimizer" : the desired optimizer used for gradient descent step  --> Compulsory , <dtype: String or Keras.optimizers>,
            "cost" : the cost function of the Network --> Compulsory  <dtype : string or Keras.losses>,
            "metrics" : The metrics to watch over during training --> Compulsory <dtype : List of strings>,
            "scaler": Determines the scaler used for preprocessing <dtype: Sklearn.scalers>,
            "save_scaler: Boolean flag determines whether to save the scaler , Default is False ---> optional <dtype: Bool>,
            "save_scaler_info: dictionary containing the name and the address for saving scalers --- > Mandatory when saving scalers <dtype :Dict>

         }
         1-a - "save_scaler_info" structure 
               save_scaler_info = {
                 "name" : "scaler name to be saved",
                 "address" : "Scaler address to be saved"
               }

'''





''' 
    Data Preprocessing is different 
    

'''


class RecurrentNN(MLmodels):
    def __init__(self,options,loading=False) -> None:
        super().__init__()
        if options is None and loading is False:
            print("Error : Options must be determined")
            return
        if options is not None and loading:
           print("----- loading an existing model -----")
           self.save_scaler = options["save_scaler"]
           self.save_scaler_address = options["save_scaler_info"]["address"]
           self.save_scaler_name = options["save_scaler_info"]["name"]
           self.length = options["length"]
           self.horizon = options["horizon"]
           self.features=options["features"]
           self.target=options["targets"]
        if options is not None and loading is False:
            self.df=options["df"]
            self.features=options["features"]
            self.target=options["targets"]
            self.model =Sequential()
            self.optimizer=options["optimizer"]
            self.cost=options["cost"]
            self.length = options["length"]
            self.horizon = options["horizon"]
            self.metrics=options['metrics']
            self.layers = options['layers']
            self.scaler = options["scaler"]
            self.save_scaler = options["save_scaler"]
            self.save_scaler_address = options["save_scaler_info"]["address"]
            self.save_scaler_name = options["save_scaler_info"]["name"]

            for layer in self.layers :
                self.model.add(layer)
    

    def arch(self):
        return self.model.summary()

    def fit(self,test_size,val_size,epochs=5,custom_split=None,split_args = None,batch_size=None,callbacks=None,random_states=None,shuffle=False):
        '''
            Performes data split , scaling and training with given parameters in options dictionary,

            Custom split is defined as : custom_split(X,y,val_size,test_size,split_args)

            Desired arguments can be passed by split_args 

            returns (training history , X_test , y_test) --> <dtype:tuple>
        '''
        data = self.data_preprocess(val_size,test_size,random_states=random_states,shuffle=shuffle,custom_split=custom_split,split_args=split_args)

        train_gen = TimeseriesGenerator(data["X_train"],data["y_train"],length=self.length,batch_size=self.horizon)
        val_gen = TimeseriesGenerator(data["X_val"],data["y_val"],length=self.length,batch_size=self.horizon)
        test_gen = TimeseriesGenerator(data["X_test"],data["y_test"],length=self.length,batch_size=self.horizon)
        model = self.model

        model.compile(optimizer= self.optimizer , loss = self.cost , metrics=self.metrics)

        train_history = model.fit(train_gen,batch_size=batch_size,validation_data=val_gen,epochs=epochs,callbacks=callbacks)
        
        return (train_history,test_gen)

    def save(self,path=None,name="Dummymodel"):
        ## for saving the model we need to save its scaler and inference related information in  a JSON file
        options = {
            "features" : self.features,
            "targets":self.target,
            "length" : self.length,
            "horizon" : self.horizon,
            "save_scaler" : 1 if self.scaler else 0,
             "save_scaler_info":{
                 "name" : self.save_scaler_name,
                 "address" : self.save_scaler_address
               }
        }
        if path is None:
            print("Save path is not determined")
            return
        saving_path = os.path.join(path,name)
        # Saving options as a json file 
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        with open(saving_path+f"/{name}.json",'w') as js:
            json.dump(options,js)
        save_address = os.path.join(saving_path,name+".h5")
        self.model.save(save_address,save_format="h5")
        print(f"------ Model is saved under \n \n {os.path.abspath(saving_path)} ")

    @classmethod
    def load(cls,path=None,name=None):
        if (path and name) :
            # Loading related Json 
            loading_path = os.path.join(path,name)
            if not os.path.exists(loading_path):
                print("Model Does not exist")
                return
            with open(loading_path+f"/{name}.json",'r') as js:
                 options = json.load(js)  
            model_address = os.path.join(loading_path,name+".h5")
            obj = MLmodels.define("RNN",options=options,loading=True)
            obj.model=load_model(model_address)
            # Loading related scalers
            if obj.save_scaler:
                obj.input_scaler = pickle.load(open(os.path.join(options["save_scaler_info"]["address"],options["save_scaler_info"]["name"]+"_input.pkl"),'rb'))
                obj.target_scaler = pickle.load(open(os.path.join(options["save_scaler_info"]["address"],options["save_scaler_info"]["name"]+"_target.pkl"),'rb'))
                obj.scaler = obj.input_scaler

            return obj
        else:
            print("Save path is not determined")
            return

    #### Predict method is overrided since RNN prediction is in sequenced based and different than supervised learning

    def predict_multi_timeseries_varying_inputs(self,first_batch,X_test):
        '''
            This function is designed for infering from RNNs with multi timeseries where independent features have varying values through out each time series

            X_test is the array showing the independent features [Independent 1 , Independent 2  ] shape(timeseries,n_independent_features )
        '''
        model = self.model 
        n_features = len(self.features)
        n_target = len(self.target)        
        out = np.zeros((X_test.shape[0],len(n_target)))

        current_batch = first_batch.reshape(1,self.length,n_features)
        temp_out = []
        ## varying input
        for i in range(X_test.shape[0]+1):
            pred = model.predict(current_batch,verbose=False)[0]
            temp_out.append(pred)
            next = np.append(X_test[i:i+self.horizon,:],temp_out[-self.horizon:]).reshape(1,self.horizon,current_batch.shape[2])
            current_batch = np.append(current_batch[:,self.horizon:,:],next,axis=1)
        out = self.target_scaler.inverse_transform(np.array(temp_out))
        return out 
    def predict_multi_timeseries_fixed_inputs(self,first_batch,steps):

        '''
            This function is designed for infering from RNNs with multi timeseries where independent features have fixed values through out each time series
        '''
        model = self.model 
        n_features = len(self.features)
        n_target = len(self.target)
        out = np.zeros((steps,n_target))
        first_batch = self.input_scaler.transform(first_batch)
        first_batch = first_batch.reshape(1,first_batch.shape[0],first_batch.shape[1])
        current_batch = first_batch
        temp_out = []
        for st in range(steps):
            pred = model.predict(current_batch,verbose=False)[0]
            temp_out.append(pred)
            next = np.append(current_batch[0,-self.horizon,:-n_target],temp_out[-self.horizon]).reshape(1,self.horizon,current_batch.shape[2])
            current_batch = np.append(current_batch[:,self.horizon:,:],next,axis=1)
        out = self.target_scaler.inverse_transform(np.array(temp_out))
        return out 
    
    def predict_multivarate_no_inputs(self,first_batch,steps):
        model = self.model 
        n_features = len(self.features)
        n_target = len(self.target)
        out = np.zeros((steps,n_target))
        first_batch = self.input_scaler.transform(first_batch)
        first_batch = first_batch.reshape(1,self.length,n_features)
        current_batch = first_batch
        print(current_batch.shape)

        temp_out = []
        for st in range(steps):
            pred = model.predict(current_batch,verbose=False)[0]
            temp_out.append(pred)
            current_batch = np.append(current_batch[:,self.horizon:,:],[[pred]],axis=1)
        out = self.target_scaler.inverse_transform(np.array(temp_out))
        return out 
    
    def predict(self, x_pred=None):
        if x_pred is None:
            x_pred = self.df[self.features]
        if self.scaler is not None:
            x_pred = self.input_scaler.transform(x_pred)
            x_pred = x_pred.reshape(1,self.length,len(self.features))
            pred = self.model.predict(x_pred)[0]
            if pred.ndim == 1 :
                pred = pred.reshape(-1,1)
            return self.target_scaler.inverse_transform(pred)
        else:
            x_pred = x_pred.reshape(1,self.length,len(self.features))
            return self.model.predict(x_pred)[0]