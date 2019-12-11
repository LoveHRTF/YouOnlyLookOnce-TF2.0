"""
Original Yolo model in Tensorflow 2.0
This model is the complete Yolo model for performing detection task
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, LeakyReLU, Dropout, Reshape
import config as cfg

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        # Hyper Parameters

        # Update Parameters and optimizer
        self.batch_size = cfg.common_params['batch_size']
        self.learning_rate = 2e-4
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Trainable Parameters
        self.model = tf.keras.Sequential(name='YouOnlyLookOnce')

        # Conv2D Block 1
        self.model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Conv2D Block 2
        self.model.add(Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Conv2D Block 3
        self.model.add(Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Conv2D Block 4
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Conv2D Block 5
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[2, 2], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))

        # Conv2D Block 6
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))

        # Flatten Layer 1
        self.model.add(Flatten())

        # Dense Layer 1, (4096, 0)
        self.model.add(Dense(4096))
        self.model.add(LeakyReLU(alpha=0.1))

        # Dense Layer 2, (7, 7, 30)
        self.model.add(Dense(4410, activation='linear'))

        # Dropout when train
        self.model.add(Dropout(0.5))

        # Reshape
        self.model.add(Reshape([7, 7, 90]))

        self.model.build()

        print(self.model.summary())

    def call(self, inputs):
        """
        Param inputs: Image matrix of [448, 448, 3]
        Return      : Predictions tensor of [7, 7, 30]
        """
        return self.model(inputs)

    def loss(self, logits, labels):
        """
        Return: Average loss. Tensor of shape [1,]
        """
        loss =  self.localizationloss(logits, labels) + \
                self.confidenceloss(logits, labels) + \
                self.classificationloss(logits, labels)

        return loss / self.batch_size

    def localizationloss(self, logits, labels):
        """
        nn_output:  (bs, 7, 7, 30)
        labels:     (bs, 7, 7, 30)
        Return: xy loss and wh loss. Tensor of shape [1,]
        """
        # Localization Loss
        coord_scale = cfg.common_params['coord_scale']
        xy      = tf.gather(logits, indices=[0,1,5,6], axis=3)
        xy_hat  = tf.gather(labels, indices=[0,1,5,6], axis=3)
        wh      = tf.gather(logits, indices=[2,3,7,8], axis=3)
        wh_hat  = tf.gather(labels, indices=[2,3,7,8], axis=3)
        loss    = coord_scale * tf.reduce_sum(tf.math.square(xy - xy_hat)) + \
                  coord_scale * tf.reduce_sum(tf.math.square(wh - wh_hat))
        return loss

    def confidenceloss(self, logits, labels):
        """
        Return: Box confidence loss. Tensor of shape [1,]
        """
        # Confidence Loss
        # noobject_scale = cfg.common_params['noobject_scale']    # not used
        c       = tf.gather(logits, indices=[4,9], axis=3)
        c_hat   = tf.gather(labels, indices=[4,9], axis=3)
        one     = (c_hat + 1.) / 2.   # confidence -> noobject scale (1->1, 0->0.5)
        loss    = tf.reduce_sum(tf.multiply(tf.square(c - c_hat), one))
        return loss

    def classificationloss(self, logits, labels):
        """
        Return: Class prob loss. Tensor of shape [1,]
        """
        # Classification Loss
        idx     = tf.range(start=10, limit=90, delta=1)
        p       = tf.gather(logits, indices=idx, axis=3)
        p_hat   = tf.gather(labels, indices=idx, axis=3)
        loss    = tf.reduce_sum(tf.math.square(p - p_hat))
        return loss

    def accuracy(self, logits, labels):
        # TODO: Add evaluation of accuracy

        pass
