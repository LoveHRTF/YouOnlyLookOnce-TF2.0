"""
Original Yolo model in Tensorflow 2.0
This model is the complete Yolo model for performing detection task
"""

import tensorflow as tf
import tensorflow.keras.layers.Conv2D as Conv2D
import tensorflow.keras.layers.MaxPool2D as MaxPool2D
import tensorflow.keras.layers.Dense as Dense


class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()
        # Hyper Parameters
        # Update Parameters and optimizer
        self.learning_rate = 1e-4
        self.optimizer.Adam(self.learning_rate)

        # Trainable Parameters
        # Conv2D Layers
        self.conv2d_1       = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding='same', activation='relu')

        self.conv2d_2       = Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')

        self.conv2d_3_1     = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_3_2     = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_3_3     = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_3_4     = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')

        self.conv2d_4_1_1   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_2_1   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_1_2   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_2_2   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_1_3   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_2_3   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_1_4   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_2_4   = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_3     = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_4_4     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')

        self.conv2d_5_1_1   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_5_2_1   = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_5_1_2   = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_5_2_2   = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_5_3     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')
        self.conv2d_5_4     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu')

        self.conv2d_6_1     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same', activation=None)
        self.conv2d_6_2     = Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same', activation=None)

        # Max Pooling Layers
        self.maxPool_1      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_2      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_3      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_4      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.maxPool_5      = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        # Fully Connected Layers
        self.dense_1        = Dense(4096, activation='relu')
        self.dense_2        = Dense([7, 7, 30], activation=None)                    # Linear activation in call

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

        dense_1_out     = self.dense_1(conv2d_6_out)                                # Dense Layer 1, (4096, 0)
        dense_1_out     = self.dropout_1(dense_1_out)

        dense_2_out     = self.dense_2(dense_1_out)                                 # Dense Layer 2, (7, 7, 30)

        output          = tf.keras.activation.linear(dense_2_out)                   # Apply linear activation layer

        return output

    def loss(self, nn_output, labels, anchors, num_class, iou_threshold):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Add loss function

        classificationLoss  = self.classificationLoss()
        localizationLoss    = self.localizationLoss()
        confidenceLoss      = self.confidenceLoss()

        loss = classificationLoss + localizationLoss + confidenceLoss
        return loss

    def classificationloss(self):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Classification Loss

        loss = None
        return loss

    def localizationloss(self):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Localization Loss

        loss = None
        return loss

    def confidenceloss(self):
        """
        Return: Tensor of shape [1,]
        """
        # TODO: Confidence Loss

        loss = None
        return loss

    def accuracy(self, logits, labels):
        # TODO: Add evaluation of accuracy

        pass
