import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
import sys
import yaml
from src.logger import logging
from src.exception import CustomException
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report
from mlflow.pyfunc import PythonModel

class CustomRandomForestModel(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def read_params(config_path):
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    
    except Exception as e:
        logging.error('Some exception happened while reading the parameters from the yaml file.')
        raise CustomException(e, sys)

def accuracymeasures(y_test, predictions, avg_method):
    """
    This function has been defined to measure all the Performance matrices.
    Input: True data & Predicted Data 
    Output: Different matrix scores, Confusion Matrix, and Classification Report
    """
    try:
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average=avg_method)
        recall = recall_score(y_test, predictions, average=avg_method)
        f1score = f1_score(y_test, predictions, average=avg_method)
        target_names = ['0', '1']

        print("Classification report")
        print("---------------------", "\n")
        print(classification_report(y_test, predictions, target_names=target_names), "\n")

        print("Confusion Matrix")
        print("---------------------", "\n")
        print(confusion_matrix(y_test, predictions), "\n")

        print("Accuracy Measures")
        print("---------------------", "\n")
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1score)

        logging.info(f"We have received the scores --> accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1score: {f1score}")
        return accuracy, precision, recall, f1score
    
    except Exception as e:
        logging.error('Some exception occurred while checking different accuracy scores of the Model')
        raise CustomException(e, sys)

def get_feat_and_target(df, target):
    """
    Get features and target variables separately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    try:
        x = df.drop(target, axis=1)
        y = df[target].values.ravel()
        return x, y
    
    except Exception as e:
        raise CustomException(e, sys) 

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth = config["random_forest"]["max_depth"]
    min_samples_leaf = config["random_forest"]["min_samples_leaf"]
    min_samples_split = config["random_forest"]["min_samples_split"]
    n_estimators = config["random_forest"]["n_estimators"]

    try:
        train = pd.read_csv(train_data_path, sep=",")
        test = pd.read_csv(test_data_path, sep=",")
        train_x, train_y = get_feat_and_target(train, target)
        test_x, test_y = get_feat_and_target(test, target)
        logging.info('We have successfully segregated the Independent & Dependent variables')

    except Exception as e:
        logging.error('We have received some error while segregating dependent and independent features')
        raise CustomException(e, sys)

    ################### MLFLOW ###############################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    try:
        with mlflow.start_run():
            model = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split, n_estimators=n_estimators)
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            accuracy = accuracy_score(test_y, y_pred)
            precision = precision_score(test_y, y_pred, average="weighted")
            recall = recall_score(test_y, y_pred, average="weighted")
            f1score = f1_score(test_y, y_pred, average="weighted")

            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("n_estimators", n_estimators)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1score)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=mlflow_config["registered_model_name"])
            else:
                mlflow.sklearn.log_model(model, "model")

    except Exception as e:
        logging.error('Some error occurred while running the experiments using MLFLOW')
        raise CustomException(e, sys)
 
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)