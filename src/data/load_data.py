import yaml
import argparse
import sys
import numpy as np 
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

def read_params(config_path):
    """
    Created a fun to read the Parameters from params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    
    except Exception as e:
        logging.error('Some exception happened while reading the parameters from the yaml file.')
        raise CustomException(e,sys)

def load_data(data_path,model_var):
    """
    In order to load the CSV dataset from the given path
    input: CSV path 
    output: Pandas Dataframe 
    """
    logging.info('Data ingestion process has begun')
    try:
        df = pd.read_csv(data_path, sep=",", encoding='utf-8')
        df=df[model_var]
        logging.info('Data ingestion process has been completed')
        return df

    except Exception as e:
        logging.error('Exception occured while reading the dataset')
        raise CustomException(e,sys)

def load_raw_data(config_path):
    """
    load data from external location(data/external) to the raw folder(data/raw) with train and testing dataset 
    input: config_path 
    output: save train file in data/raw folder 
    """
    try:
        logging.info('We have started reading the parameters from params.yaml file')
        config= read_params(config_path)

        external_data_path=config["external_data_config"]["external_data_csv"]
        raw_data_path=config["raw_data_config"]["raw_data_csv"]
        model_var=config["raw_data_config"]["model_var"]

        logging.info('We have started exporting all the chosen features and its values to the raw folder')
        df=load_data(external_data_path,model_var)
        
        df.to_csv(raw_data_path,index=False)
        logging.info('We have successfully exported the dataset from external to raw folder')

    except Exception as e:
        logging.error('Some exception has happened while exporting the data from external to raw folder')
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)