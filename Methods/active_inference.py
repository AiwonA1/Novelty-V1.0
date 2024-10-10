import numpy as np
import logging
from sklearn.linear_model import BayesianRidge
from typing import Optional
import joblib

# Configure logging for ActiveInference
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[ActiveInference] %(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ActiveInference:
    """Class to handle active inference processes using probabilistic models and Bayesian inference."""

    def __init__(self):
        # Initialize probabilistic model (e.g., Bayesian Ridge)
        self.model = BayesianRidge()
        self.current_beliefs = np.array([])
        logger.info("ActiveInference initialized with Bayesian Ridge model.")

    def update_beliefs(self, new_data: np.ndarray):
        """
        Update beliefs based on new incoming data using Bayesian updating.

        Args:
            new_data (np.ndarray): New evidence to update the model.
        """
        if self.current_beliefs.size == 0:
            self.current_beliefs = new_data
            self.model.fit(new_data.reshape(-1, 1), new_data)
            logger.info("Initial beliefs established and model trained.")
        else:
            self.model.fit(new_data.reshape(-1, 1), new_data)
            self.current_beliefs = self.model.predict(new_data.reshape(-1, 1))
            logger.info("Beliefs updated using Bayesian inference.")

    def predict_next(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict the next state based on current beliefs.

        Args:
            input_data (np.ndarray): Current input data.

        Returns:
            np.ndarray: Predicted next state.
        """
        prediction = self.model.predict(input_data.reshape(-1, 1))
        logger.info("Next state predicted using active inference.")
        return prediction

    def save_model(self, filepath: str):
        """
        Save the current Bayesian model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Bayesian model saved to {filepath}.")

    def load_model(self, filepath: str):
        """
        Load a Bayesian model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.model = joblib.load(filepath)
        logger.info(f"Bayesian model loaded from {filepath}.")

    def reset_beliefs(self):
        """
        Reset the current beliefs and model state.
        """
        self.current_beliefs = np.array([])
        self.model = BayesianRidge()
        logger.info("Beliefs and Bayesian model have been reset.")