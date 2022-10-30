import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from bowel.data.transformers.data_transformer import DataTransformer


class AudioFeaturesTransformer(DataTransformer):
    """Creates the following audio features: mfcc, rms."""

    def __init__(self, data: pd.DataFrame, config: dict):
        super().__init__(data, config)
        self._store_absolute()
        self._mfccs: np.ndarray
        self._rmss: np.ndarray

    def _transform(self):
        self._extract_features()
        self._swap_axis()
        self._concatenate()
        self._reshape_data()

    def _store_absolute(self):
        """Add absolute values to the config."""
        sr = self._config["sr"]
        self._config["fft"] = int(self._config["relative_fft"] * sr)
        self._config["hop_length"] = int(self._config["relative_hop_length"] * sr)
        self._config["noverlap"] = int(self._config["fft"] - self._config["hop_length"])
        self._config["frame_length"] = int(self._config["relative_frame_length"] * sr)

    def _extract_features(self):
        sr = self._config["sr"]
        hop_length = self._config['hop_length']
        fmax = self._config['max_freq']
        window = self._config['window_type']
        n_fft = self._config["fft"]
        n_mfcc = self._config["n_mfcc"]
        mfccs = []
        rmss = []
        for audio in tqdm(self._data):
            mfcc = librosa.feature.mfcc(audio,
                                        sr=sr,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length,
                                        n_fft=n_fft,
                                        window=window,
                                        fmax=fmax)
            rms = librosa.feature.rms(audio,
                                      frame_length=n_fft,
                                      hop_length=hop_length)
            mfccs.append(mfcc)
            rmss.append(rms)
        self._mfccs = np.asarray(mfccs)
        self._rmss = np.asarray(rmss)

    def _swap_axis(self):
        """Moves features to the last dimension."""
        self._mfccs = self._mfccs.swapaxes(1, 2)
        self._rmss = self._rmss.swapaxes(1, 2)

    def _concatenate(self):
        """Concatenates all the features into one np.ndarray."""
        self._transformed = np.concatenate([self._mfccs, self._rmss], axis=2)

    def _determine_shape(self):
        """
        Determines the required shape for the neural network input.

        It varies due to the different audio transformation configuration e.g. n_fft and hop_length.
        """
        n_substeps = int((self._config["wav_sample_length"] - self._config["relative_frame_length"]) / self._config[
            "relative_frame_length"]) + 1
        n_steps = self._transformed.shape[1]
        n_frames = n_steps // n_substeps
        shape = [self._transformed.shape[0], n_substeps, n_frames, self._transformed.shape[-1]]
        return shape

    def _reshape_data(self):
        """
        Reshapes the data into the format needed in the neural network, which is:
        (None, timesteps, subtimesteps, features) or (None, timesteps, features).
        """
        data_shape = self._determine_shape()
        n_steps = data_shape[1] * data_shape[2]
        self._transformed = self._transformed[:, :n_steps]
        if self._config["subtimesteps"]:
            self._transformed = self._transformed.reshape([data_shape[0], data_shape[1], data_shape[2], data_shape[3]])
        else:
            self._transformed = self._transformed.reshape([data_shape[0] * data_shape[1], data_shape[2], data_shape[3]])
