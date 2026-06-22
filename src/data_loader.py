"""
data_loader.py

Handles loading the Fashion-MNIST dataset via Keras' built-in dataset API.
Demonstrates the @property pattern for controlled, fail-fast attribute access.
"""

from typing import List, Optional
import sys

import numpy as np
import tensorflow as tf

from src.logger import Logger
from src.exception import CustomException

_logger_obj = Logger("data_loader")
logger = _logger_obj.get_logger()


class DataLoader:
    """
    Loads and exposes the Fashion-MNIST dataset.

    Design choice: __init__ does NOT load data immediately. It only sets up
    empty placeholders. Data is loaded explicitly via load_data(). Accessing
    any data property before load_data() has been called raises a
    CustomException — converting a silent bug (forgetting to load) into an
    immediate, debuggable error.
    """

    CLASS_NAMES: List[str] = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    def __init__(self) -> None:
        """Initialize empty placeholders. No data is loaded yet."""
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._x_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._is_loaded: bool = False
        logger.info("DataLoader instance created. Data not loaded yet.")

    def load_data(self) -> None:
        """
        Loads Fashion-MNIST via tf.keras.datasets. Downloads automatically
        on first run (cached locally by Keras afterward).
        """
        try:
            logger.info("Loading Fashion-MNIST dataset via Keras...")
            (x_train, y_train), (x_test, y_test) = (
                tf.keras.datasets.fashion_mnist.load_data()
            )
            self._x_train, self._y_train = x_train, y_train
            self._x_test, self._y_test = x_test, y_test
            self._is_loaded = True
            logger.info(
                f"Dataset loaded. Train shape: {x_train.shape}, "
                f"Test shape: {x_test.shape}"
            )
        except Exception as e:
            logger.error("Failed to load Fashion-MNIST dataset.")
            raise CustomException(e, sys)

    def _check_loaded(self) -> None:
        """Raises CustomException if data hasn't been loaded yet."""
        if not self._is_loaded:
            error_msg = (
                "Attempted to access data before calling load_data(). "
                "Call loader.load_data() first."
            )
            logger.error(error_msg)
            try:
                raise CustomException(ValueError(error_msg), sys)
            except Exception as e:
                raise CustomException(e,sys)
        

    @property
    def x_train(self) -> np.ndarray:
        """Training images, shape (60000, 28, 28)."""
        self._check_loaded()
        return self._x_train

    @property
    def y_train(self) -> np.ndarray:
        """Training labels, shape (60000,)."""
        self._check_loaded()
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        """Test images, shape (10000, 28, 28)."""
        self._check_loaded()
        return self._x_test

    @property
    def y_test(self) -> np.ndarray:
        """Test labels, shape (10000,)."""
        self._check_loaded()
        return self._y_test

    @property
    def class_names(self) -> List[str]:
        """Maps label index (0-9) to human-readable class name."""
        return self.CLASS_NAMES


if __name__ == "__main__":
    loader = DataLoader()

    try:
        print(loader.x_train)  # should raise CustomException
    except CustomException as e:
        print("--- Expected failure case ---")
        print(f"str(e):  {e}")
        print(f"repr(e): {repr(e)}")

    print("\n--- Now loading data properly ---")
    loader.load_data()
    print("Train shape:", loader.x_train.shape)
    print("Test shape:", loader.x_test.shape)
    print(
        "Sample label:", loader.y_train[0],
        "->", loader.class_names[loader.y_train[0]]
    )