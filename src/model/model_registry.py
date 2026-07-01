

# register model
import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI — make sure this matches your EC2
mlflow.set_tracking_uri("http://3.110.185.110:5000")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        logger.error('Make sure dvc repro has run successfully and experiment_info.json exists')
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug('Registering model from URI: %s', model_uri)

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.debug('Model registered as version %s', model_version.version)

        # MLflow 3.x uses aliases instead of stages
        # transition_model_version_stage() is removed in MLflow 3.x
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",              # alias replaces "Staging" stage
            version=model_version.version
        )

        logger.debug(
            f'Model {model_name} version {model_version.version} '
            f'registered and alias set to "staging".'
        )

        # return version for reference
        return model_version.version

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'experiment_info.json'

        # check file exists before loading
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(
                f'{model_info_path} not found. '
                f'Run dvc repro first to generate this file.'
            )

        model_info = load_model_info(model_info_path)
        logger.debug('Model info: %s', model_info)

        model_name = "youtube_chrome_plugin_model"
        version = register_model(model_name, model_info)
        print(f"Model '{model_name}' version {version} registered with alias 'staging'")

    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()