from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger


class DataTransformer(ABC):
    """Abstract class that transforms audio data into a features that the model will be fed with."""

    def __init__(self, data: pd.DataFrame, config: dict):
        """
        Args:
            data: pd.DataFrame index with "kfold" and "filename" index pointing to single channel audio-file
            config: transformation config with parameters needed to apply it (specific for each subclass)
                e.g. fft, hop_length

        """
        self._data = data
        self._config = config
        self._transformed = None

    @abstractmethod
    def _transform(self):
        raise NotImplementedError

    @property
    def transformed(self):
        """Returns the transformed data if available otherwise transforms it first and then returns."""
        if self._transformed is None:
            logger.info("Data transformation started")
            self._transform()
            logger.info("Data transformation done")
            return self._transformed
        else:
            return self._transformed
