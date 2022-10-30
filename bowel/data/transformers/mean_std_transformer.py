import numpy as np

from bowel.data.transformers.raw_audio_transformer import RawAudioTransformer


class MeanStdTransformer(RawAudioTransformer):
    """Creates audio mean and standard deviation features."""
    def __init__(self, data, config_path):
        super().__init__(data, config_path)

    def _transform(self):
        super(MeanStdTransformer, self)._transform()
        means = np.mean(self._transformed, axis=-1, keepdims=True)
        stds = np.std(self._transformed, axis=-1, keepdims=True)
        self._transformed = np.concatenate([means, stds], axis=-1)
