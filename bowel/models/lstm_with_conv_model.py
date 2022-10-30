import tensorflow as tf
import keras.layers as layers


class LSTM_with_conv_model(tf.keras.Model):

    def __init__(self):
        super(LSTM_with_conv_model, self).__init__()
        self._conv2d_1 = layers.TimeDistributed(layers.Conv2D(filters=8, kernel_size=2))
        self._conv2d_2 = layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=2))
        self._flatten = layers.TimeDistributed(layers.Flatten())
        self._lstm_1 = layers.LSTM(64, return_sequences=True)
        self._lstm_2 = layers.LSTM(64, return_sequences=True)
        self._dropout_1 = layers.TimeDistributed(layers.Dropout(0.4))
        self._dense = layers.Dense(100, activation='sigmoid')
        self._classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self._conv2d_1(inputs)
        x = self._conv2d_2(x)
        x = self._flatten(x)
        x = self._lstm_1(x)
        x = self._lstm_2(x)
        x = self._dropout_1(x)
        x = self._dense(x)
        return self._classifier(x)
