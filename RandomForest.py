import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from .MLModel import MLmodels
import joblib
import os
import pickle
import json
'''
    svm class is a specific extension of Support Vector Regressor.
    This class immplements one the case studies of model-informed approach. this case study is designed to observe the impact of Support Vectors 
    as a shallow learning method in model-informed method.
    INIT : 
    
       1-  The "Options" parameter has the structure bellow , this parameter is the mandatory configuration of these class.
        
        Options = {
            "df" : Pandas based dataframe of the whole dataset --> Compulsory <dtype : Pandas.DataFrame>
            "features" : data labels for the SVR (X) --> Compulsory <dtype : list of strings> ,
            "targets" : the targets for SVR (Y) --> Compulsory <dtype : list of strings> ,
            "train_size" : this variable show the index that splits training set from the test(pure prediction test) --> Compulsory <dtype : Int>,
            "n_estimators" : number of estimators designated for random forest -->Compulsory <dtype:Int>,
            "GridCVSearch" : this variable is for determining the usage of GridCvSearch method for hypemeter tunning . --> Optional ,<dtype : Bool>,
            "params" : GridCvSearch parameters for hyperparameter tunning. this is used if GridCvSearch initial parameter is passed True --> Optional <dtype : dic> must be passed when using GridCvSearch ,
            "scaler": Determines the scaler used for preprocessing <dtype: Sklearn.scalers>,
            "save_scaler: Boolean flag determines whether to save the scaler , Default is False ---> optional <dtype: Bool>,
            "save_scaler_info: dictionary containing the name and the address for saving scalers --- > Mandatory when saving scalers <dtype :Dict>
         }
    2- save_scaler_info : {"address": , "name": }
    
    3- load_options = {
            
            "load_scaler: Boolean flag determines whether to load the scaler , Default is False ---> optional <dtype: Bool>,

            "load_scaler_info: dictionary containing the name and the address for loading scalers --- > Mandatory when saving scalers <dtype :Dict>

         }
'''
class RandomForest(MLmodels):
    def __init__(self,options=None,loading=False):
        super().__init__()
        if options is None and loading is False:
            print("Options must be determined")
            return
        if loading:
           print("loading")
           pass
        if options is not None and loading is False:
            self.df=options["df"]
            self.features=options["features"]
            self.target=options["targets"]
            self.t_size=options["train_size"]
            self.model = None
            self.GridCVSearch = options["GridCVSearch"]
            self.scaler = options["scaler"]
            self.save_scaler = options["save_scaler"]
            self.save_scaler_address = options["save_scaler_info"]["address"]
            self.save_scaler_name = options["save_scaler_info"]["name"]

            if self.GridCVSearch :
                self.params= options["params"]
            else :
                self.n_estimators = options["n_estimators"]

    def fit(self,val_size,test_size,random_states,shuffle=False,custom_split = False ,split_args=None):
        data = self.data_preprocess(val_size,test_size, random_states, shuffle,custom_split,split_args)
        X_train = data['X_train']
        y_train = data['y_train']
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = data['X_test']
        y_test = data['y_test']
        self.data = data
        if self.GridCVSearch :
            grid = GridSearchCV(RandomForestRegressor(), param_grid=self.params)
            grid.fit(X_train, y_train.ravel())
            self.model = grid.best_estimator_
            # return svm_model.predict(self.df[self.features])
        else:
            model = RandomForestRegressor(n_estimators=self.n_estimators)
            model.fit(X_train,y_train.ravel())
            self.model=model
        return (self.model,X_test,y_test)

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
        joblib_file = os.path.join(path,name+".pkl")
        joblib.dump(self.model,joblib_file)
        print(f"------ Model is saved under \n \n {os.path.abspath(saving_path)} ")

    @classmethod
    def load(cls,path=None,name=None,options=None):
        if (path and name) :
            # Loading related Json 
            loading_path = os.path.join(path,name)
            if not os.path.exists(loading_path):
                print("Model Does not exist")
                return
            with open(loading_path+f"/{name}.json",'r') as js:
                 options = json.load(js)  
            joblib_file = os.path.join(path,name+".pkl")
            obj = MLmodels.define("svm",options=None,loading=True)
            obj.model=joblib.load(joblib_file)
            # Loading related scalers
            if obj.save_scaler:
                obj.input_scaler = pickle.load(open(os.path.join(options["save_scaler_info"]["address"],options["save_scaler_info"]["name"]+"_input.pkl"),'rb'))
                obj.target_scaler = pickle.load(open(os.path.join(options["save_scaler_info"]["address"],options["save_scaler_info"]["name"]+"_target.pkl"),'rb'))
            return obj
        else:
            print("Save path is not determined")
            return



