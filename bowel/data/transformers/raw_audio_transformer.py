from scipy.signal import butter, lfilter

from bowel.data.transformers.data_transformer import DataTransformer


class RawAudioTransformer(DataTransformer):
    """Filters audio using a low-pass filter or leaves it without a change (depending on the max_freq param)."""
    def __init__(self, data, config):
        super().__init__(data, config)
        self._max_freq = self._config["max_freq"]

    def _transform(self):
        self._config["frame_length"] = int(self._config["relative_frame_length"] * self._config["sr"])
        if self._max_freq is not None:
            filter_b, filter_a = butter(4, self._max_freq, "lowpass", fs=self._config["sr"])
            shape = self._data.shape
            data = self._data.reshape(shape[0], -1)
            self._data = lfilter(filter_b, filter_a, data).reshape(shape)
        if self._config["subtimesteps"]:
            self._transformed = self._data.reshape(self._data.shape[0], -1, self._config["frame_length"])
        else:
            self._transformed = self._data.reshape(-1, self._config["frame_length"])
