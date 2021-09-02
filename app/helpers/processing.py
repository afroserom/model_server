import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np
from category_encoders.woe import WOEEncoder
import warnings
from datetime import datetime
import pickle

def replace_values_in_string(text, args_dict):
    for key in args_dict.keys():
        text = text.replace(key, str(args_dict[key]))
    return text

class ModifiedColumnTransformer(ColumnTransformer):       
    """Wraps a modified version of a ColumnTransformer that includes the column names after having done all the
    transformations.
        
    Args:
        transformers (list): List of transformers that are going to be set for the ColumnTransformer inheriting parent
        numeric_features (list): List of strings containing the standard numeric features contained in the initial 
            dataset
        categorical_features (list): List of strings containing the standard categorical features contained in the
            initial dataset
        special_features (list): List of strings containing the special features contained in the
            initial dataset (could be numeric or categorical, the difference is that they get a different treatment
            than the rest in the pipeline)
        hard_mode (bool): Wheter to enforce initial fitted features during transformation or not
    Returns:
        None.
    Raises:
        None.
    """
    def __init__(self, transformers, numeric_features:list = [], categorical_features:list = [], special_features:list = [], hard_mode:bool = True):
        super().__init__(transformers=transformers)
        self.initial_features = numeric_features + categorical_features + special_features
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.special_features = special_features
        self.final_features = None
        self.hard_mode = hard_mode
        if len(self.initial_features) == 0:
            warnings.warn(f"{datetime.now()} INFO: No initial features were set, please set explicitly numeric_features, categorical_features, and/or special_features to avoid unexpected beahaviors. You can continue like this but some problems may appear when using the transformer.", stacklevel=2)
        warnings.warn(f"""{datetime.now()} INFO: Hard mode for the ModifiedColumnTransformer set to {self.hard_mode}: The initial features {'are' if self.hard_mode else 'are not'} going to be enforced during transformation and fit steps""", stacklevel=2)
    
    def fit(self, X, y=None, **kwargs):
        if self.hard_mode:
            super().fit(X[self.initial_features], y=y)
        else:
            super().fit(X, y=y)
            self.initial_features = X.columns
        self.final_features = ModifiedColumnTransformer.get_all_column_names(self)
        
    def transform(self, X, y=None):
        if self.hard_mode:
            return super().transform(X[self.initial_features])
        else:
            return super().transform(X)
        
    def fit_transform(self, X, y):
        if self.hard_mode:
            result = super().fit_transform(X[self.initial_features], y=y)
        else:
            result = super().fit_transform(X, y=y)
            self.initial_features = X.columns
        self.final_features = ModifiedColumnTransformer.get_all_column_names(self)
        return result
    
    @staticmethod
    def get_all_column_names(column_transformer) -> list:
        """Extracts the name of the resulting columns of a ColumnTransformer after all the transformations
        Args:
            column_transformer (ColumnTranformer): ColumnTransformer fitted instance from which to extract the column
                names
        Returns:
            col_name (list): List containing the column names based on the order of the ColumnTransformer transformers
        Raises:
            None.
        """
        col_name = []
        for transformer_in_columns in column_transformer.transformers_:
            # print(transformer_in_columns)
            raw_col_name = transformer_in_columns[2]
            if isinstance(transformer_in_columns[1],Pipeline): 
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]
            try:
                category_dict = {}
                i=0
                names = transformer.get_feature_names()
                for category in transformer_in_columns[2]:
                    category_dict[f"x{i}"] = category
                    i+=1
                names = [replace_values_in_string(name,category_dict) for name in names]
                # print(category_dict)
            except AttributeError: # if no 'get_feature_names' function, use raw column name
                names = raw_col_name
            if isinstance(names,np.ndarray): # eg.
                col_name += names.tolist()
            elif isinstance(names,list):
                col_name += names    
            elif isinstance(names,str):
                col_name.append(names)
        return col_name   