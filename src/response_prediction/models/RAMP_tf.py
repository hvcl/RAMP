import numpy as np
import scipy
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import argparse
import math


class Drug_Response_Predictor(keras.Model):
    def __init__(self, nparameter, ndrugs, alpha, dp_rate):
        '''
        Official Tensorflow Code for RAMP: Response-Aware Multi-task Learning with Contrastive Regularization for Cancer Drug Response Prediction
        https://github.com/hvcl/RAMP
        '''
        super(Drug_Response_Predictor, self).__init__()
        self.np = nparameter
        self.ndrugs = ndrugs
        self.h1 = layers.Dense(self.np, activation=tf.nn.leaky_relu)
        self.h2_1 = layers.Dense(self.np, activation=tf.nn.leaky_relu)
        self.h2_2 = layers.Dense(self.np, activation=tf.nn.leaky_relu)
        self.h3_1 = layers.Dense(self.np, activation=tf.nn.leaky_relu)
        self.h3_2 = layers.Dense(self.np, activation=tf.nn.leaky_relu)
        self.h4_1 = layers.Dense(ndrugs, activation=None)
        self.h4_2 = layers.Dense(ndrugs, activation=None)
        self.alpha = alpha
        self.dp_rate = dp_rate
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_tracker.reset_states()

    def call(self, inputs, training=True):
        h = self.h1(inputs)
        h1 = self.h2_1(h)
        skip_1 = h1
        if self.dp_rate != 0.0 and training:
            h1 = tf.nn.dropout(h1, rate=self.dp_rate)
        h1 = self.h3_1(h1)
        if self.dp_rate != 0.0 and training:
            h1 = tf.nn.dropout(h1, rate=self.dp_rate)
        h1 = self.h4_1(h1) # batch_size, n_drugs
    
        h2 = self.h2_2(h)
        skip_2 = h2
        if self.dp_rate != 0.0 and training:
            h2 = tf.nn.dropout(h2, rate=self.dp_rate)
        h2 = self.h3_2(h2)
        if self.dp_rate != 0.0 and training:
            h2 = tf.nn.dropout(h2, rate=self.dp_rate)
        h2 = self.h4_2(h2) # batch_size, n_drugs
        return tf.concat([h1,h2], axis=1), tf.concat([h, skip_1, skip_2], axis=1)

    def train_step(self, data):
        inputs, label = data
        label, mask = tf.split(label, 2, axis=-1)
        label = tf.squeeze(label)
        mask = tf.squeeze(mask)
        with tf.GradientTape() as tape:
            out, features = self(inputs)
            features_normalized = tf.math.l2_normalize(features, axis=1) # Batch_size, feature_size 
            logits = tf.linalg.matmul(features_normalized, tf.transpose(features_normalized))
            y_true = tf.linalg.matmul(label,label, transpose_b=True, a_is_sparse=True, b_is_sparse=True)
            y_true /= 265
            #contrastive_loss = self.alpha*tf.math.reduce_mean(-y_true*tf.math.log(logits))
            contrastive_loss = self.alpha*self.ce(y_ture, logits)
            ce_loss = tf.reduce_mean(mask*tf.nn.sigmoid_cross_entropy_with_logits(label, out))
            loss = ce_loss + contrastive_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"CE_loss":ce_loss, "Contrastive_loss":contrastive_loss}

    def predict(self, data, MCD=True):
        pred, features = self(data, training=MCD)
        pred = tf.nn.sigmoid(pred)
        neg, pos = tf.split(pred, 2, axis=-1)
        neg = tf.expand_dims(neg, axis=-1)
        pos = tf.expand_dims(pos, axis=-1)
        return tf.concat([neg, pos], axis=-1)
