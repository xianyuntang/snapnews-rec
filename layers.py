import tensorflow as tf
from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, BatchNormalizationV2, concatenate, Lambda, \
    Activation
from tensorflow.python.keras import backend as K


class SelfAttention(Layer):
    def __init__(self, units=512, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[2]
        self.Wq = self.add_weight(name=f'{self.name}_q', shape=(feature_dim, self.units)
                                  )
        self.Wk = self.add_weight(name=f'{self.name}_k', shape=(feature_dim, self.units)
                                  )
        self.Wv = self.add_weight(name=f'{self.name}_v', shape=(feature_dim, 1))

        self.bh = self.add_weight(name=f'{self.name}_bh', shape=(self.units,))

        self.ba = self.add_weight(name=f'{self.name}_ba', shape=(1,))

    def call(self, inputs, **kwargs):
        batch_size, input_len, _ = inputs.shape
        q = K.expand_dims(K.dot(inputs, self.Wq), 2)
        k = K.expand_dims(K.dot(inputs, self.Wk), 1)
        h = tf.tanh(q + k + self.bh)

        e = K.dot(h, self.Wv) + self.ba
        # e = K.reshape(e, shape=(batch_size, input_len, input_len))
        e = tf.reshape(e,shape=(batch_size,input_len,input_len))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())
        v = K.batch_dot(a, inputs)
        return v