from abc import abstractmethod, ABC


class Normalizer(ABC):
    def __init__(self, normalize=True):
        self._normalize = normalize

    @abstractmethod
    def normalize(self, data):
        raise NotImplementedError()

    @abstractmethod
    def adapt(self, data):
        raise NotImplementedError()
