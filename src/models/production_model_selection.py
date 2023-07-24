import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    
    mlflow.set_tracking_uri(remote_server_uri)

    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=mlflow.get_experiment_by_name(mlflow_config["experiment_name"]).experiment_id)
        
        # Filter runs that have the accuracy metric logged
        runs_with_accuracy = [r for r in runs if 'accuracy' in r.data.metrics]
        
        # Find the run with the maximum accuracy
        if runs_with_accuracy:
            best_run = max(runs_with_accuracy, key=lambda r: r.data.metrics['accuracy'])
            best_run_id = best_run.info.run_id
            
            # Load the model from the best run
            model_uri = f"runs:/{best_run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Set the stage of the best model to "Production" and other models to "Staging"
            for mv in client.search_model_versions(f"name='{mlflow_config['registered_model_name']}'"):
                mv = dict(mv)
                if mv["run_id"] == best_run_id:
                    current_version = mv["version"]
                    client.transition_model_version_stage(
                        name=mlflow_config["registered_model_name"],
                        version=current_version,
                        stage="Production"
                    )
                else:
                    current_version = mv["version"]
                    client.transition_model_version_stage(
                        name=mlflow_config["registered_model_name"],
                        version=current_version,
                        stage="Staging"
                    )
            
            # Save the best model to disk in the specified directory
            model_dir = config["model_dir"]
            joblib.dump(model, model_dir)
        
        else:
            print("No runs found with accuracy metric. Please make sure the 'accuracy' metric is logged during training.")

    except Exception as e:
        print("Error occurred while loading the production model:", e)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)