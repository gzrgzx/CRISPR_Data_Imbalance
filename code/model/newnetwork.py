import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,BatchNormalization,Conv1D,Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling,RandomUniform
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
from keras_layer_normalization import LayerNormalization
from tensorflow.keras.initializers import glorot_normal
import pandas as pd
import time
import shutil
import tensorflow_addons as tfa
from keras_bert import get_custom_objects
# from tensorflow.keras.layers.embeddings import Embedding
# from keras.initializers import RandomUniform
from tensorflow.python.keras.layers.core import Reshape, Permute
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply
from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten
# import keras
from transformers import BertTokenizer, TFBertModel
from vit_tensorflow.vit_3 import VisionTransformer
# from einops.layers.tensorflow import Rearrange
# from vit_tensorflow.deepvit import DeepViT
from tensorflow.python.ops import array_ops

VOCAB_SIZE = 16
EMBED_SIZE = 90
MAXLEN = 23
seed = 123

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

def plotPrecisionRecallCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			pre, rec, _ = precision_recall_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol],
				pos_label=icol)
		else:
			pre, rec, _ = precision_recall_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol],
				pos_label=icol)
		#
		plt.plot(
			rec, pre,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(rec, pre), 3))
		)
		indx += 1
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(flnm)

def plotRocCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			fprs, tprs, _ = roc_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol]
			)
		else:
			fprs, tprs, _ = roc_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol]
			)
		# print(estimator)
		plt.plot(
			fprs, tprs,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(fprs, tprs), 3))
		)
		indx += 1
	#
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.legend(loc='best')
	plt.savefig(flnm)

def transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def pre_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], 1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    x = x.astype('float32')
    print('xtrain shape:', x.shape)
    print(x.shape[0], 'train samples')

    y = to_categorical(y, num_classes)
    return x, y, input_shape

def ppre_transformIO(xtrain, ytrain, xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, ytrain,xval,yval, input_shape

def CRISPR_Net_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def pre_CRISPR_Net_transformIO(x,y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], 1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    x = x.astype('float32')

    print('xtrain shape:', x.shape)
    print(x.shape[0], 'train samples')


    y = to_categorical(y, num_classes)
    return x,y, input_shape

def ppre_CRISPR_Net_transformIO(xtrain, ytrain,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, ytrain, xval,yval, input_shape

def cnn_std_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'xval samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def new_transformIO(xtrain, xtest, ytrain, ytest,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], seq_len, coding_dim,1)
    xtest = xtest.reshape(xtest.shape[0],seq_len, coding_dim,1)
    input_shape = (seq_len, coding_dim,1)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape

def offt_transformIO(xtrain, xtest, ytrain, ytest ,xval,yval, num_classes):
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval

def pre_offt_transformIO(x, y, num_classes):
    x = x.astype('float32')
    print('xtrain shape:', x.shape)
    print(x.shape[0], 'train samples')

    y = to_categorical(y, num_classes)
    return x, y

def ppre_offt_transformIO(xtrain, ytrain, xval,yval, num_classes):
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain,  ytrain, xval,yval

def pencil_transformIO(label,xtrain, xtest, ytrain, ytest, xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    xtest = xtest.astype('float32')
    if label == 0:
        print('xtrain shape:', xtrain.shape)
        print(xtrain.shape[0], 'train samples')
    if label == 1:
        print('xtest shape:', xtest.shape)
        print(xtest.shape[0], 'test samples')
    if label == 2:
        print('xval shape:', xval.shape)
        print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def guideseq_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], seq_len, coding_dim,1)
    input_shape = (seq_len, coding_dim,1)
    x = x.astype('float32')
    print('x shape:', x.shape)
    print(x.shape[0], 'samples')

    y = to_categorical(y, num_classes)

    return x, y, input_shape

def guideseq_pencil_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], 1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    x = x.astype('float32')
    print('x shape:', x.shape)
    print(x.shape[0], 'samples')

    y = to_categorical(y, num_classes)

    return x, y, input_shape

def cnn_transformIO(xtrain, xtest, ytrain, ytest, seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],seq_len, coding_dim)
    input_shape = (seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape

def pre_cnn_std_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], 1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    x = x.astype('float32')
    print('xtrain shape:', x.shape)
    print(x.shape[0], 'train samples')

    y = to_categorical(y, num_classes)
    return x, y, input_shape

def ppre_cnn_std_transformIO(xtrain, ytrain, xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xval.shape[0], 'xval samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, ytrain, xval,yval, input_shape


class Focal_loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=10, **kwargs):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        sigmoid_p = tf.nn.sigmoid(y_pred)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.math.log(
            tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)

class dice_loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(dice_loss, self).__init__()

    def dice(self, y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def call(self, y_true, y_pred):
        return 1 - self.dice(y_true, y_pred)

class GHM_Loss(tf.keras.losses.Loss):
    def __init__(self, bins=10, momentum=0.75, **kwargs):
        super(GHM_Loss, self).__init__()
        self.g =None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-3
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
        return edges_left, edges_right


    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.compat.v1.assign(self.acc_sum,
                               tf.where(valid_bins, alpha * self.acc_sum + (1 - alpha) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot

    def call(self, y_true, y_pred, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(y_true, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(y_pred) - y_true) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(y_true)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true*train_mask,
                                                                 logits=y_pred)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_initializer=glorot_normal(seed=seed),
                      name=name, trainable=trainable)(x)

    # x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def pre_CRISPR_Net_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_undersampling.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_undersampling.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_NearMiss.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_NearMiss.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_NearMiss_2(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_NearMiss_2.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_NearMiss_2.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_NearMiss_2.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_NearMiss_3.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_NearMiss_3.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_TomekLinks.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_TomekLinks.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_SMOTETomek.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_SMOTETomek.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_SMOTEENN.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_SMOTEENN.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_ENN.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_ENN.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_oversampling.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, activation='relu')(blstm_out)
        x = Dense(20, activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2, activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_oversampling.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_model(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_smote.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_smote.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_smote.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_adasyn.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_adasyn.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_adasyn.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_BorderlineSMOTE.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_KMeansSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_KMeansSMOTE.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_KMeansSMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_KMeansSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net_SVMSMOTE.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80,activation='relu')(blstm_out)
        x = Dense(20,activation='relu')(x)
        x = Dropout(0.35)(x)
        prediction = Dense(2,activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_Net_SVMSMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_Net_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=Focal_loss(alpha=alpha, gamma=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CRISPR_Net_undersampling_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_Net_undersampling_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=Focal_loss(alpha=alpha, gamma=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_Net_undersampling_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_Net_undersampling_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CRISPR_Net_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_Net_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=GHM_Loss(bins=alpha, momentum=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_Net_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_Net_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model


def pre_CRISPR_IP_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_oversampling.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_oversampling.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model


def pre_CRISPR_IP_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_SMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_SMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_SMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_ADASYN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_ADASYN.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_ADASYN.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_BorderlineSMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_SVMSMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_SVMSMOTE.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_undersampling.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_undersampling.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_NearMiss.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_NearMiss.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_NearMiss_3.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_NearMiss_3.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_TomekLinks.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_TomekLinks.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_ENN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_ENN.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_SMOTETomek.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_SMOTETomek.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP_SMOTEENN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP_SMOTEENN.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_model(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_IP.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(
            tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_IP.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_IP.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_IP_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_IP_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                bidirectional_1_output = Bidirectional(
                    LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                    Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
                attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
                average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
                max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
                concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                flatten_output = Flatten()(concat_output)
                linear_1_output = BatchNormalization()(
                    Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
                linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.9)(linear_2_output)
                linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=Focal_loss(alpha=alpha, gamma=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_IP_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_IP_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CRISPR_IP_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_IP_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                bidirectional_1_output = Bidirectional(
                    LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                    Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
                attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
                average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
                max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
                concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                flatten_output = Flatten()(concat_output)
                linear_1_output = BatchNormalization()(
                    Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
                linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.9)(linear_2_output)
                linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=GHM_Loss(bins=alpha, momentum=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_IP_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_IP_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, x):
        # print('x:')
        # print(x.shape)
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        # print('position_embedding')
        # print(position_embedding.shape)
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        # print((position_embedding+x).shape)
        return position_embedding+x
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_len' : self.sequence_len,
            'embedding_dim' : self.embedding_dim,
        })
        return config


# branch_0 = conv2d_bn(inputs, 10, (1, 1))
#         branch_1 = conv2d_bn(inputs, 10, (1, 2))
#         branch_2 = conv2d_bn(inputs, 10, (1, 3))
#         branch_3 = conv2d_bn(inputs, 10, (1, 5))
#         branches = [inputs, branch_0, branch_1, branch_2, branch_3]
#         # branches = [branch_0, branch_1, branch_2, branch_3]
#         mixed = Concatenate(axis=-1)(branches)
#         mixed = Reshape((23, 46))(mixed)
#         blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
#         # inputs_rs = Reshape((24, 7))(inputs)
#         # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
#         blstm_out = Flatten()(blstm_out)


def pre_CrisprDNT_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_oversampling.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)

        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_oversampling.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_oversampling.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_model(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CrisprDNT_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, input_shape[-1]))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=Focal_loss(alpha=alpha, gamma=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CrisprDNT_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding, 'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_undersampling_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CrisprDNT_undersampling_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, input_shape[-1]))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=Focal_loss(alpha=alpha, gamma=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CrisprDNT_undersampling_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding, 'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_undersampling_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CrisprDNT_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, input_shape[-1]))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=GHM_Loss(bins=alpha, momentum=gamma),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CrisprDNT_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding, 'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_SMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_SMOTE.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_SMOTE.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_ADASYN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_ADASYN.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_ADASYN.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_BorderlineSMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_BorderlineSMOTE.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_SVMSMOTE.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_SVMSMOTE.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_SVMSMOTE.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_undersampling.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_undersampling.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_undersampling.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_NearMiss.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_NearMiss.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_NearMiss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_NearMiss_3.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_NearMiss_3.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_NearMiss_3.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_TomekLinks.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_TomekLinks.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_TomekLinks.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_ENN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_ENN.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_ENN.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_SMOTETomek.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_SMOTETomek.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_SMOTETomek.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CrisprDNT_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CrisprDNT_SMOTEENN.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                               data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23, input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CrisprDNT_SMOTEENN.h5'.format(saved_prefix))
    else:
        # model = load_model('{}+CrisprDNT.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'PositionalEncoding': PositionalEncoding}
        custom_objects.update(my_objects)
        model = load_model('{}+CrisprDNT_SMOTEENN.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CNN_std_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CNN_std_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(1, 23, 4))
                conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
                conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
                conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
                conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

                conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

                bn_output = BatchNormalization()(conv_output)

                pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

                flatten_output = Flatten()(pooling_output)

                x = Dense(100, activation='relu')(flatten_output)
                x = Dense(23, activation='relu')(x)
                x = Dropout(rate=0.15)(x)

                output = Dense(2, activation="softmax")(x)

                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CNN_std_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CNN_std_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CNN_std_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CNN_std_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(1, 23, 4))
                conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
                conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
                conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
                conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

                conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

                bn_output = BatchNormalization()(conv_output)

                pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

                flatten_output = Flatten()(pooling_output)

                x = Dense(100, activation='relu')(flatten_output)
                x = Dense(23, activation='relu')(x)
                x = Dropout(rate=0.15)(x)

                output = Dense(2, activation="softmax")(x)

                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CNN_std_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CNN_std_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_cnn_std(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_SMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_SMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_SMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_ADASYN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_ADASYN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_ADASYN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_BorderlineSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_SVMSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_SVMSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_oversampling.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_oversampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_undersampling.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_undersampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_cnn_std_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_NearMiss.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_NearMiss.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_NearMiss_3.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_NearMiss_3.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_cnn_std_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_TomekLinks.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_TomekLinks.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_cnn_std_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_ENN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_ENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_cnn_std_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_SMOTETomek.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_SMOTETomek.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_cnn_std_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std_SMOTEENN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+cnn_std_SMOTEENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_dl_crispr(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_SMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_SMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_SMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_ADASYN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_ADASYN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_ADASYN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_BorderlineSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_SVMSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_SVMSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_oversampling.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_oversampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_dl_crispr_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(1, 20, 20))
                conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
                conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
                conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
                conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

                flatten_output = Flatten()(conv_4)

                x = Dense(100, activation='relu')(flatten_output)

                output = Dense(2, activation="linear")(x)

                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+dl_crispr_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+dl_crispr_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_dl_crispr_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(1, 20, 20))
                conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
                conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
                conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
                conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

                flatten_output = Flatten()(conv_4)

                x = Dense(100, activation='relu')(flatten_output)

                output = Dense(2, activation="linear")(x)

                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+dl_crispr_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+dl_crispr_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model


def pre_dl_crispr_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_undersampling.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_undersampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_dl_crispr_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_NearMiss.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_NearMiss.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_NearMiss_3.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_NearMiss_3.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_TomekLinks.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_TomekLinks.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_dl_crispr_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_ENN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_ENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_SMOTETomek.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_SMOTETomek.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_dl_crispr_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+dl_crispr_SMOTEENN.h5'.format(saved_prefix)):
        input = Input(shape=(1, 20, 20))
        conv_1 = Conv2D(20, (8, 8), strides=2, padding='same', activation='relu')(input)
        conv_2 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(40, (2, 2), strides=1, padding='same', activation='relu')(conv_3)

        flatten_output = Flatten()(conv_4)

        x = Dense(100, activation='relu')(flatten_output)

        output = Dense(2, activation="linear")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+dl_crispr_SMOTEENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+dl_crispr_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_CnnCrispr(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_SMOTE(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_SMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_SMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_SMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_ADASYN(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_ADASYN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_ADASYN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_ADASYN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_BorderlineSMOTE(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_BorderlineSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_SVMSMOTE(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_SVMSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_SVMSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_oversampling(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_oversampling.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_oversampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_focal_loss(embedding_weights,xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(23,))
                embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                                      weights=[embedding_weights],
                                      trainable=True)(input)
                bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
                bi_lstm_relu = Activation('relu')(bi_lstm)

                conv1 = Conv1D(10, (5))(bi_lstm_relu)
                conv1_relu = Activation('relu')(conv1)
                conv1_batch = BatchNormalization()(conv1_relu)

                conv2 = Conv1D(20, (5))(conv1_batch)
                conv2_relu = Activation('relu')(conv2)
                conv2_batch = BatchNormalization()(conv2_relu)

                conv3 = Conv1D(40, (5))(conv2_batch)
                conv3_relu = Activation('relu')(conv3)
                conv3_batch = BatchNormalization()(conv3_relu)

                conv4 = Conv1D(80, (5))(conv3_batch)
                conv4_relu = Activation('relu')(conv4)
                conv4_batch = BatchNormalization()(conv4_relu)

                conv5 = Conv1D(100, (5))(conv4_batch)
                conv5_relu = Activation('relu')(conv5)
                conv5_batch = BatchNormalization()(conv5_relu)

                flat = Flatten()(conv5_batch)
                drop = Dropout(0.3)(flat)
                dense = Dense(20)(drop)
                dense_relu = Activation('relu')(dense)
                prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
                model = Model(input, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy'])
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CnnCrispr_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CnnCrispr_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CnnCrispr_GHM(embedding_weights, xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(23,))
                embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                                      weights=[embedding_weights],
                                      trainable=True)(input)
                bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
                bi_lstm_relu = Activation('relu')(bi_lstm)

                conv1 = Conv1D(10, (5))(bi_lstm_relu)
                conv1_relu = Activation('relu')(conv1)
                conv1_batch = BatchNormalization()(conv1_relu)

                conv2 = Conv1D(20, (5))(conv1_batch)
                conv2_relu = Activation('relu')(conv2)
                conv2_batch = BatchNormalization()(conv2_relu)

                conv3 = Conv1D(40, (5))(conv2_batch)
                conv3_relu = Activation('relu')(conv3)
                conv3_batch = BatchNormalization()(conv3_relu)

                conv4 = Conv1D(80, (5))(conv3_batch)
                conv4_relu = Activation('relu')(conv4)
                conv4_batch = BatchNormalization()(conv4_relu)

                conv5 = Conv1D(100, (5))(conv4_batch)
                conv5_relu = Activation('relu')(conv5)
                conv5_batch = BatchNormalization()(conv5_relu)

                flat = Flatten()(conv5_batch)
                drop = Dropout(0.3)(flat)
                dense = Dense(20)(drop)
                dense_relu = Activation('relu')(dense)
                prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
                model = Model(input, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy'])
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CnnCrispr_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CnnCrispr_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CnnCrispr_undersampling(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_undersampling.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_undersampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_NearMiss(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_NearMiss.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_NearMiss.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_NearMiss_3(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_NearMiss_3.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_NearMiss_3.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_TomekLinks(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_TomekLinks.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_TomekLinks.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_ENN(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_ENN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_ENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_SMOTETomek(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_SMOTETomek.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_SMOTETomek.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CnnCrispr_SMOTEENN(embedding_weights,test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr_SMOTEENN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                              weights=[embedding_weights],
                              trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CnnCrispr_SMOTEENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CnnCrispr_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_CRISPR_OFFT(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_SMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_SMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_SMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_SMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_ADASYN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_ADASYN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_ADASYN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_ADASYN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_BorderlineSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_BorderlineSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_BorderlineSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_BorderlineSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_SVMSMOTE(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_SVMSMOTE.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_SVMSMOTE.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_SVMSMOTE.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model


def pre_CRISPR_OFFT_oversampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_oversampling.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_oversampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_oversampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_focal_loss(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_focal_loss.h5'.format(saved_prefix)):
        alpha_list = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        gamma_list = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 30, 50, 60, 80, 100]
        # alpha_list = [0.8]
        # gamma_list = [10.0]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(23,))
                embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

                conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
                batchnor1 = BatchNormalization()(conv1)

                conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
                batchnor2 = BatchNormalization()(conv2)

                conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
                batchnor3 = BatchNormalization()(conv3)

                conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
                x = Attention()([conv11, batchnor3])

                flat = Flatten()(x)
                dense1 = Dense(40, activation="relu", name="dense1")(flat)
                drop1 = Dropout(0.2)(dense1)

                dense2 = Dense(20, activation="relu", name="dense2")(drop1)
                drop2 = Dropout(0.2)(dense2)

                output = Dense(2, activation="softmax", name="dense3")(drop2)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_OFFT_focal_loss.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        return best_model
    else:
        # model = load_model('{}+CRISPR_Net_focal_loss.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
        custom_objects = get_custom_objects()
        my_objects = {'Focal_loss': Focal_loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_OFFT_focal_loss.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model

def pre_CRISPR_OFFT_GHM(xtest, ytest, test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_GHM.h5'.format(saved_prefix)):
        alpha_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        gamma_list = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9]
        # alpha_list = [10]
        # gamma_list = [0.75]
        best_pr = 0
        best_roc = 0
        best_gamma = 0
        best_alpha = 0
        best_model = 0
        for alpha in alpha_list:
            for gamma in gamma_list:
                print(alpha)
                print(gamma)
                input = Input(shape=(23,))
                embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

                conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
                batchnor1 = BatchNormalization()(conv1)

                conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
                batchnor2 = BatchNormalization()(conv2)

                conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
                batchnor3 = BatchNormalization()(conv3)

                conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
                x = Attention()([conv11, batchnor3])

                flat = Flatten()(x)
                dense1 = Dense(40, activation="relu", name="dense1")(flat)
                drop1 = Dropout(0.2)(dense1)

                dense2 = Dense(20, activation="relu", name="dense2")(drop1)
                drop2 = Dropout(0.2)(dense2)

                output = Dense(2, activation="softmax", name="dense3")(drop2)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy'])  # Adam是0.001，SGD是0.01
                history_model = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=[xval, yval],
                    # steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            precision, recall, thresholds = precision_recall_curve(gcetest, yscore)
                            score = np.round(auc(recall, precision), 4)
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_gamma = gamma
                                best_model = model
                                model.save('{}+CRISPR_OFFT_GHM.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


        return best_model
    else:
        custom_objects = get_custom_objects()
        my_objects = {'GHM_Loss': GHM_Loss}
        custom_objects.update(my_objects)
        model = load_model('{}+CRISPR_OFFT_GHM.h5'.format(saved_prefix),
                           custom_objects=custom_objects)
    return model


def pre_CRISPR_OFFT_undersampling(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_undersampling.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_undersampling.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_undersampling.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_NearMiss(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_NearMiss.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_NearMiss.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_NearMiss.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_NearMiss_3(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_NearMiss_3.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_NearMiss_3.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_NearMiss_3.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_TomekLinks(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_TomekLinks.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_TomekLinks.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_TomekLinks.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_ENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_ENN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_ENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_ENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_SMOTETomek(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_SMOTETomek.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_SMOTETomek.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_SMOTETomek.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def pre_CRISPR_OFFT_SMOTEENN(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+CRISPR_OFFT_SMOTEENN.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=[xval, yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('{}+CRISPR_OFFT_SMOTEENN.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+CRISPR_OFFT_SMOTEENN.h5'.format(saved_prefix))
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model