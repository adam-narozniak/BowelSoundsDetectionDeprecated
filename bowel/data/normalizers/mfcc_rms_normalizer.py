import numpy as np
from keras import layers

from bowel.data.normalizers.normalizer import Normalizer


class MfccRmsNormalizer(Normalizer):
    def __init__(self, normalize=True, mean_mfcc=None, variance_mfcc=None, mean_rms=None, variance_rms=None):
        super().__init__(normalize)
        self._normalize = normalize
        self.mfcc_normalizer = layers.Normalization(axis=None, mean=mean_mfcc, variance=variance_mfcc)
        self.rms_normalizer = layers.Normalization(axis=None, mean=mean_rms, variance=variance_rms)

    def adapt(self, data):
        if self._normalize is False:
            pass
        else:
            self.mfcc_normalizer.adapt(data[:, :, :, :-1])
            self.rms_normalizer.adapt(data[:, :, :, -1])

    def normalize(self, data):
        if self._normalize is False:
            return data
        else:
            normalized_mfcc = self.mfcc_normalizer(data[:, :, :, :-1])
            normalized_rms = np.expand_dims(self.rms_normalizer(data[:, :, :, -1]), axis=-1)
            return np.concatenate([normalized_mfcc, normalized_rms], axis=-1)
