"""
Original Yolo model in Tensorflow 2.0
This model is the complete Yolo model for performing detection task
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
import config as cfg

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        # Hyper Parameters
        # Update Parameters and optimizer
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.LeakyReLU = LeakyReLU(alpha=0.1)

        # Trainable Parameters
        # Conv2D Layers

        self.conv2d_1       = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding='same', activation=self.LeakyReLU)

        self.conv2d_2       = Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)

        self.conv2d_3_1     = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_3_2     = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_3_3     = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_3_4     = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)

        self.conv2d_4_1_1   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_2_1   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_1_2   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_2_2   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_1_3   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_2_3   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_1_4   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_2_4   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_3     = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_4_4     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)

        self.conv2d_5_1_1   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_5_2_1   = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_5_1_2   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_5_2_2   = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_5_3     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=self.LeakyReLU)
        self.conv2d_5_4     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=self.LeakyReLU)

        self.conv2d_6_1     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same', activation=self.LeakyReLU)
        self.conv2d_6_2     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same', activation=self.LeakyReLU)

        # Max Pooling Layers
        self.maxPool_1      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_2      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_3      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_4      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_5      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        # Flatten Layer
        self.flat_1         = Flatten()

        # Fully Connected Layers
        self.dense_1        = Dense(4096, activation=self.LeakyReLU)
        self.dense_2        = Dense(1470, activation=None)                    # Linear activation in call

        # Dropout Layers
        self.dropout_1      = tf.keras.layers.Dropout(0.5)                          # Drop out to prevent overfit

    def call(self, inputs):
        """
        Param inputs: Image matrix of [448, 448, 3]
        Return      : Predictions tensor of [7, 7, 30]
        """
        # Model Layers

        conv2d_1_out    = self.conv2d_1(inputs)                                     # Conv2d Block 1
        maxPool_1_out   = self.maxPool_1(conv2d_1_out)

        conv2d_2_out    = self.conv2d_2(maxPool_1_out)                              # Conv2d Block 2
        maxPool_2_out   = self.maxPool_2(conv2d_2_out)

        conv2d_3_out    = self.conv2d_3_1(maxPool_2_out)                            # Conv2d Block 3
        conv2d_3_out    = self.conv2d_3_2(conv2d_3_out)
        conv2d_3_out    = self.conv2d_3_3(conv2d_3_out)
        conv2d_3_out    = self.conv2d_3_4(conv2d_3_out)
        maxPool_3_out   = self.maxPool_3(conv2d_3_out)

        conv2d_4_out    = self.conv2d_4_1_1(maxPool_3_out)                          # Conv2d Block 4
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

        conv2d_5_out    = self.conv2d_5_1_1(maxPool_4_out)                          # Conv2d Block 5
        conv2d_5_out    = self.conv2d_5_2_1(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_1_2(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_2_2(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_3(conv2d_5_out)
        conv2d_5_out    = self.conv2d_5_4(conv2d_5_out)

        conv2d_6_out    = self.conv2d_6_1(conv2d_5_out)                             # Conv2d Block 6
        conv2d_6_out    = self.conv2d_6_2(conv2d_6_out)

        flatten_out     = self.flat_1(conv2d_6_out)                                 # Flatten layer

        dense_1_out     = self.dense_1(flatten_out)                                 # Dense Layer 1, (4096, 0)
        dense_1_out     = self.dropout_1(dense_1_out)

        dense_2_out     = self.dense_2(dense_1_out)                                 # Dense Layer 2, (7, 7, 30)

        output          = tf.keras.activation.linear(dense_2_out)                   # Apply linear activation layer

        return tf.reshape(output, [-1, 7, 7, 30])                                   # Reshape to [batch_size, 7, 7, 30]

    def loss(self, logits, labels):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Add loss function

        classificationLoss  = self.classificationLoss()
        localizationLoss    = self.localizationLoss()
        confidenceLoss      = self.confidenceLoss()

        loss = localizationLoss(logits, labels) + confidenceLoss(logits, labels) + classificationLoss(logits, labels)
        return loss

    def localizationloss(self, logits, labels):
        """
        nn_output:  (bs, 7, 7, 30)
        labels:     (bs, 7, 7, 30)
        Return: Tensor of shape [1,]
        """
        # TODO: Localization Loss
        coord_scale = cfg.common_params['coord_scale']
        xy      = logits[:, :, :, [0,1,5,6]]
        xy_hat  = labels[:, :, :, [0,1,5,6]]
        wh      = tf.math.sqrt(logits[:, :, :, [2,3,7,8]])
        wh_hat  = tf.math.sqrt(labels[:, :, :, [2,3,7,8]])
        loss = coord_scale * tf.reduce_sum(tf.math.square(xy - xy_hat))
        return loss

    def confidenceloss(self, logits, labels):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Confidence Loss
        noobject_scale = cfg.common_params['noobject_scale']    # not used
        c       = logits[:, :, :, [4,9]]
        c_hat   = labels[:, :, :, [4,9]]
        one = (c_hat + 1.) / 2.   # confidence -> noobject scale (1->1, 0->0.5)
        loss = tf.reduce_sum(tf.multiply(tf.square(c - c_hat), one))
        return loss

    def classificationloss(self, logits, labels):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Classification Loss
        p       = logits[:, :, :, 10:]
        p_hat   = labels[:, :, :, 10:]
        loss = tf.reduce_sum(tf.math.square(p - p_hat))
        return loss

    def accuracy(self, logits, labels):
        # TODO: Add evaluation of accuracy

        pass
