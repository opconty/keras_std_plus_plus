#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午3:39
#
# @Author: Greg Gao(laygin)
#'''
'''adapted from csp, many thanks to the original authors'''
from keras import backend as K
import tensorflow as tf

epsilon = 1e-4


def cls_center(y_true, y_pred):
    classification_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 2])
    positives = y_true[:, :, :, 2]
    negatives = y_true[:, :, :, 1]-y_true[:, :, :, 2]
    foreground_weight = positives * (1.0 - y_pred[:, :, :, 0]) ** 2.0
    background_weight = negatives * ((1.0 - y_true[:, :, :, 0])**4.0)*(y_pred[:, :, :, 0] ** 2.0)
    focal_weight = foreground_weight + background_weight
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
    class_loss = 0.01*tf.reduce_sum(focal_weight*classification_loss) / tf.maximum(1.0, assigned_boxes)
    return class_loss


def regr_h(y_true, y_pred):
    absolute_loss = tf.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
    square_loss = 0.5 * (y_true[:, :, :, 0] - y_pred[:, :, :, 0]) ** 2
    l1_loss = y_true[:, :, :, 1]*tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 1])
    class_loss = tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
    return class_loss


def regr_offset(y_true, y_pred):
    absolute_loss = tf.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :])
    square_loss = 0.5 * (y_true[:, :, :, :2] - y_pred[:, :, :, :]) ** 2
    l1_loss = y_true[:, :, :, 2] * tf.reduce_sum(
        tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
    class_loss = 0.1*tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
    return class_loss


def cl(y_true, y_pred):
    l = K.binary_crossentropy(y_pred[:,:,:,0], y_true[:,:,:,0])
    return tf.reduce_mean(l)


# this is only for skew boxes
def angle(y_true, y_pred):
    '''maybe same as regr_h'''
    absolute_loss = tf.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
    square_loss = 0.5 * (y_true[:, :, :, 0] - y_pred[:, :, :, 0]) ** 2
    l1_loss = y_true[:, :, :, 1] * tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 1])
    class_loss = tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
    return class_loss

