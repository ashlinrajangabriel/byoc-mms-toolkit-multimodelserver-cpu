import logging
import pickle
import json
import os
import glob
import pandas as pd


class ModelHandler:
    """Handles model operations including initialization, preprocessing, inference, and postprocessing."""

    def __init__(self):
        self.initialized = False
        self.models = {}
        self.model = None  # Current active model
        self.log_file = None  # File handle for logging

    def setup_logging(self, model_dir):
        """Set up logging with a file in append mode in the specified model directory.

        Args:
            model_dir (str): Directory to set up logging file.
        """
        log_file_path = os.path.join(model_dir, "model_log.txt")
        self.log_file = open(log_file_path, "a")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=self.log_file)

    def get_model_files_prefix(self, model_dir):
        """Find a .pkl model file in the specified directory and return its prefix.

        Args:
            model_dir (str): Directory to search for model file.

        Returns:
            str: Prefix of the found model file.

        Raises:
            FileNotFoundError: If no model files are found in the directory.
        """
        model_file_pattern = os.path.join(model_dir, "*.pkl")
        model_files = glob.glob(model_file_pattern)
        if not model_files:
            raise FileNotFoundError("No model files found in the specified directory: " + model_dir)
        
        model_file = model_files[0]  # Assuming the first model file is the required one
        model_prefix = os.path.basename(model_file).split(".")[0]
        logging.info(f"Prefix for the model artifacts found: {model_prefix}")
        return model_prefix

    def load_model(self, model_name, model_dir):
        """Load a model from a specified directory using its name.

        Args:
            model_name (str): Name of the model to load.
            model_dir (str): Directory from which to load the model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No model file found for '{model_name}' at '{model_path}'")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logging.info(f"Model loaded successfully: {model_name}")

    def initialize(self, context):
        """Initialize the Model Handler by setting up logging, finding model prefix, and loading the model.

        Args:
            context (dict): Context containing system properties including model directory.

        Raises:
            Exception: Generic exception if initialization fails.
        """
        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir", "./")
            self.setup_logging(model_dir)
            model_prefix = self.get_model_files_prefix(model_dir)
            self.load_model(model_prefix, model_dir)
            self.initialized = True
            logging.info("ModelHandler initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization failure: {str(e)}")
            raise

    def preprocess(self, request):
        """Preprocess the request data to extract model input.

        Args:
            request (list): Request data containing the model input.

        Returns:
            dict: Processed data ready for inference.

        Raises:
            Exception: If preprocessing fails.
        """
        try:
            data = request
            for item in data:
                body_str = item['body'].decode('utf-8')
                body_dict = json.loads(body_str)
            return body_dict
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise

    def inference(self, model_input):
        """Perform model inference on the processed input.

        Args:
            model_input (dict): Input data for model inference.

        Returns:
            list: Model predictions.

        Raises:
            Exception: If inference fails.
        """
        try:
            predictions = self.model.predict(pd.DataFrame(model_input))
            return predictions.tolist()
        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            logging.error(error_message)
            raise

    def postprocess(self, inference_output):
        """Postprocess the inference results.

        Args:
            inference_output (list): Raw predictions from the model.

        Returns:
            list: Postprocessed model predictions.
        """
        print("just output", inference_output)
        return [inference_output]

    def handle(self, data, context):
        """Handle the full model processing from data input to response output.

        Args:
            data (list): Input data for model processing.
            context (dict): Contextual information for processing.

        Returns:
            list or dict: Processed model output or error.

        Raises:
            Exception: If handling fails at any stage.
        """
        try:
            if not self.initialized:
                raise Exception("Service has not been initialized. Please call initialize first.")

            if data is None:
                raise ValueError("No data provided for handling.")

            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            return self.postprocess(model_output)
        except Exception as e:
            logging.error(f"Error in handling the request: {str(e)}")
            return {"error": str(e)}

# Usage example
_service = ModelHandler()

def handle(data, context):
    """Global handle function to process model requests.

    Args:
        data (list): Data to be processed by the model.
        context (dict): Context for the model processing.

    Returns:
        dict: Final processed output or error.
    """
    try:
        if not _service.initialized:
            _service.initialize(context)
        return _service.handle(data, context)
    except Exception as e:
        logging.error(f"Error in global handle function: {str(e)}")
        return {"error": str(e)}
