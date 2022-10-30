from keras import layers

from bowel.data.normalizers.normalizer import Normalizer


class AllNormalizer(Normalizer):
    def __init__(self, normalize=True, mean=None, variance=None):
        super().__init__(normalize)
        self.normalizer = layers.Normalization(axis=None, mean=mean, variance=variance)

    def adapt(self, data):
        if self._normalize is False:
            pass
        else:
            self.normalizer.adapt(data)

    def normalize(self, data):
        if self._normalize is False:
            return data
        else:
            return self.normalizer(data)
