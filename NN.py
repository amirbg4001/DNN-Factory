from keras.models import Sequential,load_model
from .MLModel import MLmodels
import os
import pickle
import json
'''
    NerualNet class is a Decorator pattern for Nerual Network architecture implemmented by Tensorflow.
    Decorator design pattern solves the need of specific behaviour of an object.In this project the immplemented network is used as a the AI part
    of the model-informed model
    INIT : 

       1-  The "Options" parameter has the structure bellow , this parameter is the mandatory configuration of these class.

        Options = {
            "df" : Pandas based dataframe of the whole dataset --> Compulsory <dtype : Pandas.DataFrame>
            "features" : data labels for the SVR (X) --> Compulsory <dtype : list of strings> ,
            "targets" : the targets for SVR (Y) --> Compulsory <dtype : list of strings> ,
            "layers" : an ordered list of the Neural Network Layers and their activation function   --> Compulsory <dtype : List> ,
            "optimizer" : the desired optimizer used for gradient descent step  --> Compulsory , <dtype: String or Keras.optimizers>,
            "cost" : the cost function of the Network --> Compulsory  <dtype : string or Keras.losses>,
            "metrics" : The metrics to watch over during training --> Compulsory <dtype : List of strings>,
            "scaler": Determines the scaler used for preprocessing <dtype: Sklearn.scalers>,
            "save_scaler: Boolean flag determines whether to save the scaler , Default is False ---> optional <dtype: Bool>,
            "save_scaler_info: dictionary containing the name and the address for saving scalers --- > Mandatory when saving scalers <dtype :Dict>

         }

'''

class NerualNet(MLmodels):
    def __init__(self,options=None,loading=False):
        super().__init__()
        if options is None and loading is False:
            print("Options must be determined")
            return
        if options is not None and loading:
           print("----- loading an existing model -----")
           self.save_scaler = options["save_scaler"]
           self.save_scaler_address = options["save_scaler_info"]["address"]
           self.save_scaler_name = options["save_scaler_info"]["name"]
           self.features=options["features"]
           self.target=options["targets"]
           pass
        if options is not None and loading is False:
            self.df=options["df"]
            self.features=options["features"]
            self.target=options["targets"]
            self.model =Sequential()
            self.optimizer=options["optimizer"]
            self.cost=options["cost"]
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


    def fit(self,val_size,test_size,random_states,shuffle=False,epochs=5,batch_size=None,callbacks=None,custom_split=None,split_args = None):
        
        '''
            Performes data split , scaling and training with given parameters in options dictionary,

            Custom split is defined as : custom_split(X,y,val_size,test_size,split_args)

            Desired arguments can be passed by split_args 

            returns (training history , X_test , y_test) --> <dtype:tuple>
        '''
        data = self.data_preprocess(val_size,test_size, random_states, shuffle,custom_split,split_args)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val =data['y_val']
        model = self.model
        model.compile(optimizer=self.optimizer,loss=self.cost,metrics=self.metrics)

        if batch_size is None:
         return (model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),callbacks=callbacks),data["X_test"],data["y_test"])
        else:
         return (model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),batch_size=batch_size,callbacks=callbacks),data["X_test"],data["y_test"])

    def save(self,path=None,name="Dummymodel"):
        ## for saving the model we need to save its scaler and inference related information in  a JSON file
        options = {
            "features" : self.features,
            "targets":self.target,
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
            obj = MLmodels.define("NN",options=options,loading=True)
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

