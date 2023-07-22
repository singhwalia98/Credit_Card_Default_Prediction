import joblib
import mlflow
import logging
import argparse
import sys
from pprint import pprint
from mlflow.sklearn import load_model
from train_model import read_params
from mlflow.tracking import MlflowClient
from src.exception import CustomException
from src.logger import logging

def log_production_model(config_path, mlflow_config, max_accuracy_run_id):
    try:
        config = read_params(config_path)
        model_name = mlflow_config["registered_model_name"]
        model_dir = config["model_dir"]

        mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
        client = MlflowClient()
        logged_model = None

        for mv in client.search_model_versions(f"name='{model_name}'"):
            mv = dict(mv)
            if mv["run_id"] == max_accuracy_run_id:
                current_version = mv["version"]
                logged_model = mv["source"]
                pprint(mv, indent=4)
                client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
            else:
                current_version = mv["version"]
                client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")

        if logged_model is not None:
            model = load_model(logged_model)
            joblib.dump(model, model_dir)
            logging.info('We have found our best Model and have been dumped successfully using Joblib in the Models dir')

    except Exception as e:
        logging.error('Exception occured while choosing the Best model and dumping it in Models dir')
        raise CustomException(e,sys)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    config = read_params(parsed_args.config)
    mlflow_config = config["mlflow_config"]

    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    runs = mlflow.search_runs(experiment_ids=[1])
    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]

    log_production_model(parsed_args.config, mlflow_config, max_accuracy_run_id)