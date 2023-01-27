#Let's start with importing necessary libraries
import pickle
# import numpy as np
import pandas as pd

class predObj:

    def predict_log(self,dict_pred):
        

        with open("Dec_tree_cls_model2.pkl", 'rb') as f:
            model = pickle.load(f)

        data_df = pd.DataFrame(dict_pred,index=[1,])
        
        
        predict = model.predict(data_df)
        if predict[0] == 0:
            result = 'This person might not survive'
        else :
            result = 'This person might survive'

        return result



