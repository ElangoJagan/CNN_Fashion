"""
preprocessor.py

Prepares raw Fashion-MNIST data for CNN training:
normalization, channel-dimension reshape, and train/validation split.
"""
from typing import Optional,Tuple
import sys
import numpy as np
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger("preprocessor")
logger = _logger_obj.get_logger()

class Preprocessor:
    
    """
    Transforms raw image/label arrays into CNN-ready data.

    Design choice: normalize() and add_channel_dimension() are @staticmethod
    because they are pure transformations — given an array, they return a
    new array, without needing any instance state (self.something). They're
    grouped here for organization, not because they depend on this object.

    process() does NOT run automatically in __init__ — same fail-fast
    pattern as DataLoader. Call process() explicitly; accessing results
    before that raises CustomException.
    """
    
    def __init__(self, x_train, y_train, x_test, y_test, val_size = 0.2, random_state = 42):
        """Store raw data and split config. Nothing is processed yet."""
        self._raw_x_train= x_train
        self._raw_y_train= y_train
        self._raw_x_test= x_test
        self._raw_y_test= y_test
        self._val_size = val_size
        self._random_state = random_state
        
        
        self._x_train: Optional[np.ndarray] = None
        self._x_val: Optional[np.ndarray] = None
        self._x_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

        self._is_processed: bool = False
        logger.info("Preprocessor instance created. Data not processed yet.")
        
        @staticmethod
        def normalize(images):
            return images /255
        
        @staticmethod
        def add_channel_dimension(images):
            """Reshapes (N, 28, 28) -> (N, 28, 28, 1) for Conv2D compatibility."""
            return images.reshape(-1, 28,28,1)
        
        def process(self):
            """
            Runs the full preprocessing pipeline:
            normalize -> add channel dimension -> train/validation split.
            Test data is normalized/reshaped but NEVER split further —
            it stays untouched for final evaluation only.
            """
        try:
            logger.info("Starting preprocessing pipeline...")

            x_train_norm = self.normalize(self._raw_x_train)
            x_test_norm = self.normalize(self._raw_x_test)
            logger.info("Normalization complete (pixels scaled to 0.0-1.0).")

            x_train_reshaped = self.add_channel_dimension(x_train_norm)
            x_test_reshaped = self.add_channel_dimension(x_test_norm)
            logger.info(
                f"Channel dimension added. New train shape: "
                f"{x_train_reshaped.shape}"
            )

            x_train, x_val, y_train, y_val = train_test_split(
                x_train_reshaped,
                self._raw_y_train,
                test_size=self._val_size,
                random_state=self._random_state,
            )
            logger.info(
                f"Train/validation split done. "
                f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}"
            )

            self._x_train, self._y_train = x_train, y_train
            self._x_val, self._y_val = x_val, y_val
            self._x_test, self._y_test = x_test_reshaped, self._raw_y_test

            self._is_processed = True
            logger.info("Preprocessing pipeline completed successfully.")

        except Exception as e:
            logger.error("Preprocessing pipeline failed.")
            raise CustomException(e, sys)

    def _check_processed(self) -> None:
        """Raises CustomException if process() hasn't been called yet."""
        if not self._is_processed:
            error_msg = (
                "Attempted to access processed data before calling process(). "
                "Call preprocessor.process() first."
            )
            logger.error(error_msg)
            try:
                raise ValueError(error_msg)
            except Exception as e:
                raise CustomException(e, sys)

    @property
    def train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x_train, y_train) after processing."""
        self._check_processed()
        return self._x_train, self._y_train

    @property
    def val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x_val, y_val) after processing."""
        self._check_processed()
        return self._x_val, self._y_val

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x_test, y_test) after processing."""
        self._check_processed()
        return self._x_test, self._y_test


if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    loader.load_data()

    preprocessor = Preprocessor(
        loader.x_train, loader.y_train, loader.x_test, loader.y_test
    )

    try:
        print(preprocessor.train_data)  # should raise CustomException
    except CustomException as e:
        print("--- Expected failure case ---")
        print(f"str(e):  {e}")

    print("\n--- Now processing properly ---")
    preprocessor.process()

    x_train, y_train = preprocessor.train_data
    x_val, y_val = preprocessor.val_data
    x_test, y_test = preprocessor.test_data

    print("x_train shape:", x_train.shape, "| min/max:", x_train.min(), x_train.max())
    print("x_val shape:", x_val.shape)
    print("x_test shape:", x_test.shape)