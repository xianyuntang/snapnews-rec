import tensorflow as tf
from tensorflow.python.keras.layers import Dense, MaxPooling2D, Conv2D, \
    Reshape, Activation, BatchNormalization
from layers import SelfAttention
from tensorflow.python.keras.models import Model


class VGGATTModel(Model):
    def __init__(self, training=True):
        super(VGGATTModel, self).__init__()
        # block 1
        self.block1_conv1 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.block1_conv2 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        self.block1_batch_norm = BatchNormalization(name='block1_batch_norm')

        # block 2
        self.block2_conv1 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block2_conv2 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        self.block2_batch_norm = BatchNormalization(name='block2_batch_norm')

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block3_conv2 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block3_conv3 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block3_pool = MaxPooling2D((2, 2), strides=(1, 2), name='block3_pool')
        self.block3_batch_norm = BatchNormalization(name='block3_batch_norm')

        # Block 4
        self.block4_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block4_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block4_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block4_pool = MaxPooling2D((2, 2), strides=(1, 2), name='block4_pool')
        self.block4_batch_norm = BatchNormalization(name='block4_batch_norm')

        # Block 5
        self.blcok5_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block5_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.block5_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.block5_pool = MaxPooling2D((1, 2), strides=(1, 2), name='block5_pool')
        self.block5_batch_norm = BatchNormalization(name='block5_batch_norm')

        # Block 6
        self.block6_reshape = Reshape(target_shape=(-1, 512))
        self.self_attention1 = SelfAttention(name='attention')
        # Block 7
        self.block7_prediction = Dense(units=4651, kernel_initializer='he_normal', name='ctc_y')
        self.training = training
        if not training:
            self.block7_softmax_pred = Activation('softmax', name='softmax')


    def call(self, inputs, training=None, mask=None):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block1_batch_norm(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        x = self.block2_batch_norm(x)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        x = self.block3_batch_norm(x)
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)
        x = self.block4_batch_norm(x)
        x = self.blcok5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        x = self.block5_batch_norm(x)
        x = self.block6_reshape(x)
        x = self.self_attention1(x)
        x = self.block7_prediction(x)
        if not self.training:
            pred_text = self.block7_softmax_pred(x)
            return pred_text
        return x
