import keras
import keras.layers as layers


class LSTM_model(keras.Model):

    def __init__(self):
        super(LSTM_model, self).__init__()
        self._lstm_1 = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
        self._lstm_2 = layers.TimeDistributed(layers.LSTM(128))
        self._conv_1 = layers.Conv1D(64, kernel_size=15, activation='relu', padding='same', strides=1)
        self._classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self._lstm_1(inputs)
        x = self._lstm_2(x)
        x = self._conv_1(x)
        return self._classifier(x)
