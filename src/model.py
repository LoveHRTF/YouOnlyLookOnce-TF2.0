# Original Yolo model in Tensorflow 2.0
import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        # Hyper Parameters
        # TODO: Update Parameters and optimizer
        self.learning_rate = None
        self.optimizer.Adam(self.learning_rate)

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
        self.conv2d_5_4     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[2,2], padding='same', activation=None, bias_initializer='zeros')

        self.conv2d_6_1     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[3, 3], padding='same', activation=None, bias_initializer='zeros')
        self.conv2d_6_2     = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3,3], strides=[3, 3], padding='same', activation=None, bias_initializer='zeros')

        # Max Pooling Layers
        self.maxPool_1      = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same')
        self.maxPool_2      = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same')
        self.maxPool_3      = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same')
        self.maxPool_4      = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same')
        self.maxPool_5      = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same')

        # Fully Connected Layers
        self.dense_1        = tf.keras.layers.Dense(4096, activation=None)
        self.dense_2        = tf.keras.layers.Dense([7,7,30], activation=None)

    def call(self, inputs):
        # Model Layers
        conv2d_1_out    = self.conv2d_1(inputs)                         # Conv2d Block 1
        maxPool_1_out   = self.maxPool_1(conv2d_1_out)

        conv2d_2_out    = self.conv2d_2(maxPool_1_out)                  # Conv2d Block 2
        maxPool_2_out   = self.maxPool_2(conv2d_2_out)

        conv2d_3_out    = self.conv2d_3_1(maxPool_2_out)                # Conv2d Block 3
        conv2d_3_out    = self.conv2d_3_2(conv2d_3_out)
        conv2d_3_out    = self.conv2d_3_3(conv2d_3_out)
        conv2d_3_out    = self.conv2d_3_4(conv2d_3_out)
        maxPool_3_out   = self.maxPool_3(conv2d_3_out)

        conv2d_4_out    = self.conv2d_4_1_1(maxPool_3_out)              # Conv2d Block 4
        conv2d_4_out    = self.conv2d_4_2_1(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_1_2(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_2_2(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_1_3(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_2_3(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_1_4(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_2_4(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_3(conv2d_4_out)
        conv2d_4_out    = self.conv2d_4_4(conv2d_4_out)
        maxPool_4_out   = self.maxPool_4(conv2d_4_out)

        conv2d_5_out    = self.conv2d_5_1_1(maxPool_4_out)              # Conv2d Block 5
        conv2d_5_out    = self.conv2d_5_2_1(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_1_2(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_2_2(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_3(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_4(conv2d_5_out)

        conv2d_6_out    = self.conv2d_6_1(conv2d_5_out)                 # Conv2d Block 6
        conv2d_6_out    = self.conv2d_6_2(conv2d_6_out)

        dense_1_out     = self.dense_1(conv2d_6_out)                    # Dense Layer 1, (4096, 0)
        dense_2_out     = self.dense_2(dense_1_out)                     # Dense Layer 2, (7, 7, 30)

        return dense_2_out


    def loss(self, logits, labels):

        pass

    def accuracy(self, logits, labels):

        pass
    