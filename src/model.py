# Original Yolo model in Tensorflow 2.0
import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        # Hyper Parameters

        # Trainable Parameters
        # Conv2D Layers
        self.conv2d_1       = tf.keras.layers.Conv2D(filters=64, kernel_size=[7,7], strides=[2,2], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_2       = tf.keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_3_1     = tf.keras.layers.Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_3_2     = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_3_3     = tf.keras.layers.Conv2D(filters=256, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_3_4     = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_4_1_1   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_2_1   = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_1_2   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_2_2   = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_1_3   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_2_3   = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_1_4   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_2_4   = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_3     = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_4_4     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_5_1_1   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_5_2_1   = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_5_1_2   = tf.keras.layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_5_2_2   = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_5_3     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[1,1], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_5_3     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[2,2], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_6_1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[3, 3], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_6_2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[3, 3], padding='same', activation=None, bias_initializer='zeros')

        # Fully Connected Layers
        self.dense_1 = tf.keras.layers.Dense(4096, activation=None)
        self.dense_2 = tf.keras.layers.Dense([7,7,30], activation=None)

        pass


    def call(self, inputs):
        # Model Layers
        pass


    def loss(self, logits, labels):

        pass

    def accuracy(self, logits, labels):

        pass
    