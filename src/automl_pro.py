import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging
import os

# Configure logging for better insights into pipeline execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Pipeline:
    """
    A simplified Automated Machine Learning (AutoML) pipeline for rapid prototyping
    and deployment of ML models. Supports classification and regression tasks.
    """

    def __init__(self, task="classification", random_state=42):
        """
        Initializes the AutoML Pipeline.

        Args:
            task (str): The type of machine learning task. Can be "classification" or "regression".
            random_state (int): Seed for reproducibility.
        """
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be \'classification\' or \'regression\'.")
        self.task = task
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        logging.info(f"AutoML Pipeline initialized for {self.task} task.")

    def _preprocess_data(self, X, y):
        """
        Internal method to preprocess input data, including label encoding for classification.
        """
        # Handle categorical features in X (simple approach: one-hot encoding for demonstration)
        X = pd.get_dummies(X, drop_first=True)

        if self.task == "classification":
            # Encode target labels for classification if they are not numeric
            if y.dtype == \'object\' or y.dtype == \'category\':
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                logging.info("Target labels encoded for classification.")
        return X, y

    def fit(self, X, y):
        """
        Fits the AutoML pipeline to the training data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        logging.info("Starting model training...")
        X_processed, y_processed = self._preprocess_data(X.copy(), y.copy())

        if self.task == "classification":
            self.model = RandomForestClassifier(random_state=self.random_state, n_estimators=100, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(random_state=self.random_state, n_estimators=100, n_jobs=-1)

        self.model.fit(X_processed, y_processed)
        logging.info("Model training completed.")

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            np.array or pd.Series: Predicted values.
        """
        logging.info("Generating predictions...")
        X_processed, _ = self._preprocess_data(X.copy(), pd.Series(np.zeros(len(X)))) # Dummy y for preprocessing
        predictions = self.model.predict(X_processed)

        if self.task == "classification" and self.label_encoder:
            # Decode labels back to original format if encoding was applied
            predictions = self.label_encoder.inverse_transform(predictions)
            logging.info("Predicted labels decoded.")
        return predictions

    def evaluate(self, X, y):
        """
        Evaluates the model\'s performance.

        Args:
            X (pd.DataFrame): Feature matrix for evaluation.
            y (pd.Series): True target vector.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating model performance...")
        X_processed, y_processed = self._preprocess_data(X.copy(), y.copy())
        predictions = self.model.predict(X_processed)

        metrics = {}
        if self.task == "classification":
            metrics["accuracy"] = accuracy_score(y_processed, predictions)
            logging.info(f"Classification Accuracy: {metrics[\'accuracy\']:.4f}")
        else:
            metrics["mse"] = mean_squared_error(y_processed, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            logging.info(f"Regression RMSE: {metrics[\'rmse\']:.4f}")
        return metrics

    def deploy(self, target="local", model_path="./deployed_model.pkl"):
        """
        Simulates model deployment.

        Args:
            target (str): Deployment target (e.g., "local", "docker", "aws").
            model_path (str): Path to save the model if target is "local".
        """
        logging.info(f"Attempting to deploy model to {target}...")
        if self.model is None:
            logging.warning("No model trained yet. Please call .fit() first.")
            return

        if target == "local":
            try:
                import joblib
                joblib.dump(self.model, model_path)
                logging.info(f"Model successfully deployed locally to {model_path}")
            except ImportError:
                logging.error("joblib not installed. Cannot save model locally. Please install with \'pip install joblib\'.")
        elif target == "docker":
            logging.info("Simulating Docker containerization and deployment.")
            # In a real scenario, this would involve creating Dockerfiles, building images, etc.
            logging.info("Docker deployment simulation complete.")
        elif target == "aws":
            logging.info("Simulating deployment to AWS SageMaker.")
            # This would involve AWS SDK calls, S3 uploads, endpoint creation, etc.
            logging.info("AWS deployment simulation complete.")
        else:
            logging.warning(f"Unsupported deployment target: {target}")


if __name__ == "__main__":
    # --- Classification Example ---
    logging.info("\n--- Running Classification Example ---")
    data_clf = {
        \'feature1\': [10, 20, 15, 25, 30, 12, 22, 18, 28, 35, 11, 21, 16, 26, 31],
        \'feature2\': [\'A\', \'B\', \'A\', \'C\', \'B\', \'A\', \'C\', \'B\', \'A\', \'C\', \'B\', \'A\', \'C\', \'B\', \'A\'],
        \'feature3\': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        \'target\': [\'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\', \'Class2\', \'Class1\']
    }
    df_clf = pd.DataFrame(data_clf)
    X_clf = df_clf[[\'feature1\', \'feature2\', \'feature3\']]
    y_clf = df_clf[\'target\']

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    pipeline_clf = Pipeline(task="classification")
    pipeline_clf.fit(X_train_clf, y_train_clf)
    metrics_clf = pipeline_clf.evaluate(X_test_clf, y_test_clf)
    print(f"Classification Metrics: {metrics_clf}")
    predictions_clf = pipeline_clf.predict(X_test_clf)
    print(f"Classification Predictions: {predictions_clf}")
    pipeline_clf.deploy(target="local", model_path="./classification_model.pkl")

    # --- Regression Example ---
    logging.info("\n--- Running Regression Example ---")
    data_reg = {
        \'featureA\': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        \'featureB\': [5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52],
        \'target\': [12.5, 18.2, 25.1, 30.5, 36.8, 44.3, 50.9, 57.6, 65.2, 70.8, 77.5, 85.1, 91.7, 98.4, 105.0]
    }
    df_reg = pd.DataFrame(data_reg)
    X_reg = df_reg[[\'featureA\', \'featureB\']]
    y_reg = df_reg[\'target\']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    pipeline_reg = Pipeline(task="regression")
    pipeline_reg.fit(X_train_reg, y_train_reg)
    metrics_reg = pipeline_reg.evaluate(X_test_reg, y_test_reg)
    print(f"Regression Metrics: {metrics_reg}")
    predictions_reg = pipeline_reg.predict(X_test_reg)
    print(f"Regression Predictions: {predictions_reg}")
    pipeline_reg.deploy(target="docker")

    logging.info("\n--- AutoML Pipeline Pro examples finished ---")
