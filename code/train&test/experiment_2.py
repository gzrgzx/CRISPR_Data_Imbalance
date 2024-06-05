import pandas as pd
import numpy as np
# from codes.encoding import my_encode_on_off_dim
import newnetwork
import tensorflow as tf
import os
import sklearn
import pickle as pkl
from sklearn.model_selection import (train_test_split, GridSearchCV)
from tensorflow.python.keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import KFold

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
# import tensorflow as tf
#
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)

def loadGlove(inputpath, outputpath=""):
    data_list = []
    wordEmb = {}
    with open(inputpath) as f:
        for line in f:
            # 基本的数据整理
            ll = line.strip().split(',')
            ll[0] = str(int(float(ll[0])))
            data_list.append(ll)

            # 构建wordembeding的选项
            ll_new = [float(i) for i in ll]
            emb = np.array(ll_new[1:], dtype="float32")
            wordEmb[str(int(ll_new[0]))] = emb

    if outputpath != "":
        with open(outputpath) as f:
            for data in data_list:
                f.writelines(' '.join(data))
        # data_list = [float(i) for i in data_list]
    return wordEmb

seed = 123
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# gpus = tf.config.experimental.list_physical_devices(devices='0', device_type='GPU')
# print(os.environ['CUDA_VISIBLE_DEVICES'])
import random

random.seed(seed)

os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# print(K.image_data_format())

# Incorporating reduced learning and early stopping for NN callback
early_stopping = tf.keras.callbacks.EarlyStopping(#monitor监控的数据接口,verbose是否输出更多的调试信息
    monitor='loss', min_delta=0.0001,#0.0001
    patience=10, verbose=0, mode='auto')
callbacks = [early_stopping]

# callbacks = []

# list_dataset = ['k562','crispor','hek293t']
list_dataset = ['1&2&3&4']
test_dataset = ['Listgarten_22gRNA', 'Kleinstiver_5gRNA']
# list_dataset = ['Kleinstiver_5gRNA']
# list_dataset = ['SITE-Seq_offTarget']
# list_dataset = ['crispor']
# list_type = ['8x23','14x23']
list_type = ['14x23']
num_classes = 2
epochs = 500
batch_size = 128#64
retrain=True
flpath = '../data/'

# encoder_shape1=(9,23)
# seg_len1, coding_dim1 = encoder_shape1
# encoder_shape2=(10,23)
# seg_len2, coding_dim2 = encoder_shape2
# encoder_shape3=(13,23)
# seg_len3, coding_dim3 = encoder_shape3

from tensorflow.keras.models import model_from_json, load_model
from keras_bert import get_custom_objects

flag = 16

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

if flag == 16:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000

            print('CRISPR-OFFT model')

            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x1, y1 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_model = newnetwork.pre_CRISPR_OFFT(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR_OFFT Test')
            yscore = CRISPR_OFFT_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            prob1 = yscore
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT SMOTE')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'
            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x5, y5 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_SMOTE = newnetwork.pre_CRISPR_OFFT_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT SMOTE Test')
            yscore = CRISPR_OFFT_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            prob5 = yscore
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT ADASYN')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x6, y6 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_ADASYN = newnetwork.pre_CRISPR_OFFT_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT ADASYN Test')
            yscore = CRISPR_OFFT_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            prob6 = yscore
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x7, y7 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_BorderlineSMOTE = newnetwork.pre_CRISPR_OFFT_BorderlineSMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT BorderlineSMOTE Test')
            yscore = CRISPR_OFFT_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            prob7 = yscore
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR-OFFT SVMSMOTE')
            # # type = '14x23'
            # open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = np.array(loaddata.images), loaddata.target
            #
            # # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = SVMSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
            #     x_train, y_train, np.array(x_val), y_val, num_classes)
            #
            # x9, y9 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = False
            #
            # CRISPR_OFFT_SVMSMOTE = newnetwork.pre_CRISPR_OFFT_SVMSMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
            #                                            resampled_ds, xtrain, ytrain,
            #                                            num_classes, batch_size, epochs, callbacks,
            #                                            open_name, retrain)
            #
            # print('CRISPR-OFFT SVMSMOTE Test')
            # yscore = CRISPR_OFFT_SVMSMOTE.predict(x9)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data9 = ypred
            # yscore = yscore[:, 1]
            # prob9 = yscore
            # # print(yscore)
            # ytest = np.argmax(y9, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT oversampling')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x4, y4 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_oversampling = newnetwork.pre_CRISPR_OFFT_oversampling(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT oversampling Test')
            yscore = CRISPR_OFFT_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            prob4 = yscore
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT_focal_loss')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x2, y2 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_OFFT_focal_loss = newnetwork.pre_CRISPR_OFFT_focal_loss(x2, y2, test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       num_classes, batch_size, epochs,
                                                                       callbacks,
                                                                       open_name, retrain)

            print('CRISPR-OFFT_focal_loss Test')
            yscore = CRISPR_OFFT_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            prob2 = yscore
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT_GHM')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x3, y3 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_GHM = newnetwork.pre_CRISPR_OFFT_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         num_classes, batch_size, epochs,
                                                         callbacks,
                                                         open_name, retrain)

            print('CRISPR-OFFT_GHM Test')
            yscore = CRISPR_OFFT_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            prob3 = yscore
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT undersampling')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x10, y10 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_undersampling = newnetwork.pre_CRISPR_OFFT_undersampling(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT undersampling Test')
            yscore = CRISPR_OFFT_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            prob10 = yscore
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT NearMiss')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x11, y11 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_NearMiss = newnetwork.pre_CRISPR_OFFT_NearMiss(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT NearMiss Test')
            yscore = CRISPR_OFFT_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            prob11 = yscore
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


            print('CRISPR-OFFT NearMiss_3')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x14, y14 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_NearMiss_3 = newnetwork.pre_CRISPR_OFFT_NearMiss_3(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT NearMiss_3 Test')
            yscore = CRISPR_OFFT_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            prob14 = yscore
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT TomekLinks')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x15, y15 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_TomekLinks = newnetwork.pre_CRISPR_OFFT_TomekLinks(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT TomekLinks Test')
            yscore = CRISPR_OFFT_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            prob15 = yscore
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT ENN')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x12, y12 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_ENN = newnetwork.pre_CRISPR_OFFT_ENN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT ENN Test')
            yscore = CRISPR_OFFT_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            prob12 = yscore
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT SMOTETomek')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x16, y16 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_SMOTETomek = newnetwork.pre_CRISPR_OFFT_SMOTETomek(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT SMOTETomek Test')
            yscore = CRISPR_OFFT_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR-OFFT SMOTEENN')
            # type = '14x23'
            open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_offt_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x17, y17 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_offt_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_OFFT_SMOTEENN = newnetwork.pre_CRISPR_OFFT_SMOTEENN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CRISPR-OFFT SMOTEENN Test')
            yscore = CRISPR_OFFT_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [CRISPR_OFFT_model, CRISPR_OFFT_focal_loss, CRISPR_OFFT_GHM]

            labels = ['CRISPR-OFFT', 'CRISPR-OFFT_focal_loss', 'CRISPR-OFFT_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CRISPR-OFFT_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR-OFFT_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_OFFT_model, CRISPR_OFFT_oversampling,
                      CRISPR_OFFT_SMOTE, CRISPR_OFFT_ADASYN, CRISPR_OFFT_BorderlineSMOTE]

            labels = ['CRISPR-OFFT', 'CRISPR-OFFT_OverSampling',
                      'CRISPR-OFFT_SMOTE', 'CRISPR-OFFT_ADASYN', 'CRISPR-OFFT_KMeansSMOTE']

            xtests = [x1, x4, x5, x6, x7]

            ytests = [y1, y4, y5, y6, y7]

            roc_name = 'roccurve_imbalance_CRISPR-OFFT_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR-OFFT_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_OFFT_model, CRISPR_OFFT_undersampling, CRISPR_OFFT_NearMiss, CRISPR_OFFT_ENN,
                      CRISPR_OFFT_NearMiss_3, CRISPR_OFFT_TomekLinks]

            labels = ['CRISPR-OFFT', 'CRISPR-OFFT_UnderSampling', 'CRISPR-OFFT_NearMiss', 'CRISPR-OFFT_ENN',
                      'CRISPR-OFFT_NearMiss_2', 'CRISPR-OFFT_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CRISPR-OFFT_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR-OFFT_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_OFFT_model, CRISPR_OFFT_SMOTETomek,
                      CRISPR_OFFT_SMOTEENN]

            labels = ['CRISPR-OFFT', 'CRISPR-OFFT_SMOTETomek', 'CRISPR-OFFT_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CRISPR-OFFT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR-OFFT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CRISPR-OFFT': data1, 'CRISPR-OFFT_focal_loss': data2, 'CRISPR-OFFT_GHM': data3,
                         'CRISPR-OFFT_oversampling': data4, 'CRISPR-OFFT_SMOTE': data5, 'CRISPR-OFFT_ADASYN': data6,
                         'CRISPR-OFFT_BorderlineSMOTE': data7,
                         'CRISPR-OFFT_undersampling': data10,
                         'CRISPR-OFFT_NearMiss': data11, 'CRISPR-OFFT_ENN': data12,
                         'CRISPR-OFFT_NearMiss_3': data14, 'CRISPR-OFFT_TomekLinks': data15,
                         'CRISPR-OFFT_SMOTETomek': data16, 'CRISPR-OFFT_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CRISPR-OFFT.csv', index=None)

            data_save = {'test': ytest, 'CRISPR-OFFT': prob1, 'CRISPR-OFFT_focal_loss': prob2, 'CRISPR-OFFT_GHM': prob3,
                         'CRISPR-OFFT_oversampling': prob4, 'CRISPR-OFFT_SMOTE': prob5, 'CRISPR-OFFT_ADASYN': prob6,
                         'CRISPR-OFFT_BorderlineSMOTE': prob7,
                         'CRISPR-OFFT_undersampling': prob10,
                         'CRISPR-OFFT_NearMiss': prob11, 'CRISPR-OFFT_ENN': prob12,
                         'CRISPR-OFFT_NearMiss_3': prob14, 'CRISPR-OFFT_TomekLinks': prob15,
                         'CRISPR-OFFT_SMOTETomek': prob16, 'CRISPR-OFFT_SMOTEENN': prob17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CRISPR-OFFT_prob.csv', index=None)




if flag == 15:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000

            print('CnnCrispr model')
            print("GloVe model loaded")
            VOCAB_SIZE = 16  # 4**3
            EMBED_SIZE = 100
            glove_inputpath = "../data/keras_GloVeVec_" + dataset + "_5_100_10000.csv"
            # load GloVe model
            model_glove = loadGlove(glove_inputpath)
            embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))  # 词语的数量×嵌入的维度
            for i in range(VOCAB_SIZE):
                embedding_weights[i, :] = model_glove[str(i)]

            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x1, y1 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_model = newnetwork.pre_CnnCrispr(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr Test')
            yscore = CnnCrispr_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            prob1 = yscore
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr SMOTE')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'
            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x5, y5 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_SMOTE = newnetwork.pre_CnnCrispr_SMOTE(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr SMOTE Test')
            yscore = CnnCrispr_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            prob5 = yscore
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr ADASYN')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x6, y6 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_ADASYN = newnetwork.pre_CnnCrispr_ADASYN(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr ADASYN Test')
            yscore = CnnCrispr_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            prob6 = yscore
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x7, y7 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_BorderlineSMOTE = newnetwork.pre_CnnCrispr_BorderlineSMOTE(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr BorderlineSMOTE Test')
            yscore = CnnCrispr_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            prob7 = yscore
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CnnCrispr SVMSMOTE')
            # # type = '14x23'
            # open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = np.array(loaddata.images), loaddata.target
            #
            # # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = SVMSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
            #     x_train, y_train, np.array(x_val), y_val, num_classes)
            #
            # x9, y9 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = False
            #
            # CnnCrispr_SVMSMOTE = newnetwork.pre_CnnCrispr_SVMSMOTE(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
            #                                            resampled_ds, xtrain, ytrain,
            #                                            num_classes, batch_size, epochs, callbacks,
            #                                            open_name, retrain)
            #
            # print('CnnCrispr SVMSMOTE Test')
            # yscore = CnnCrispr_SVMSMOTE.predict(x9)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data9 = ypred
            # yscore = yscore[:, 1]
            # prob9 = yscore
            # # print(yscore)
            # ytest = np.argmax(y9, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr oversampling')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x4, y4 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_oversampling = newnetwork.pre_CnnCrispr_oversampling(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr oversampling Test')
            yscore = CnnCrispr_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            prob4 = yscore
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr_focal_loss')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x2, y2 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_focal_loss = newnetwork.pre_CnnCrispr_focal_loss(embedding_weights, x2, y2, test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       num_classes, batch_size, epochs,
                                                                       callbacks,
                                                                       open_name, retrain)

            print('CnnCrispr_focal_loss Test')
            yscore = CnnCrispr_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            prob2 = yscore
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr_GHM')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x3, y3 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_GHM = newnetwork.pre_CnnCrispr_GHM(embedding_weights, x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         num_classes, batch_size, epochs,
                                                         callbacks,
                                                         open_name, retrain)

            print('CnnCrispr_GHM Test')
            yscore = CnnCrispr_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            prob3 = yscore
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr undersampling')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x10, y10 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_undersampling = newnetwork.pre_CnnCrispr_undersampling(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr undersampling Test')
            yscore = CnnCrispr_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            prob10 = yscore
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr NearMiss')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x11, y11 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_NearMiss = newnetwork.pre_CnnCrispr_NearMiss(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr NearMiss Test')
            yscore = CnnCrispr_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            prob11 = yscore
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


            print('CnnCrispr NearMiss_3')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x14, y14 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_NearMiss_3 = newnetwork.pre_CnnCrispr_NearMiss_3(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr NearMiss_3 Test')
            yscore = CnnCrispr_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            prob14 = yscore
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr TomekLinks')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x15, y15 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_TomekLinks = newnetwork.pre_CnnCrispr_TomekLinks(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr TomekLinks Test')
            yscore = CnnCrispr_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            prob15 = yscore
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr ENN')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x12, y12 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_ENN = newnetwork.pre_CnnCrispr_ENN(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr ENN Test')
            yscore = CnnCrispr_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            prob12 = yscore
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr SMOTETomek')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x16, y16 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_SMOTETomek = newnetwork.pre_CnnCrispr_SMOTETomek(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr SMOTETomek Test')
            yscore = CnnCrispr_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CnnCrispr SMOTEENN')
            # type = '14x23'
            open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded_CnnCrispr_' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = np.array(loaddata.images), loaddata.target

            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval = newnetwork.ppre_offt_transformIO(
                x_train, y_train, np.array(x_val), y_val, num_classes)

            x17, y17 = newnetwork.pre_offt_transformIO(np.array(loaddata1.images), loaddata1.target, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded_CnnCrispr_' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CnnCrispr_SMOTEENN = newnetwork.pre_CnnCrispr_SMOTEENN(embedding_weights, test_ds, xval, yval, resampled_steps_per_epoch,
                                                       resampled_ds, xtrain, ytrain,
                                                       num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

            print('CnnCrispr SMOTEENN Test')
            yscore = CnnCrispr_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [CnnCrispr_model, CnnCrispr_focal_loss, CnnCrispr_GHM]

            labels = ['CnnCrispr', 'CnnCrispr_focal_loss', 'CnnCrispr_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CnnCrispr_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CnnCrispr_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CnnCrispr_model, CnnCrispr_oversampling,
                      CnnCrispr_SMOTE, CnnCrispr_ADASYN, CnnCrispr_BorderlineSMOTE]

            labels = ['CnnCrispr', 'CnnCrispr_OverSampling',
                      'CnnCrispr_SMOTE', 'CnnCrispr_ADASYN', 'CnnCrispr_KMeansSMOTE']

            xtests = [x1, x4, x5, x6, x7]

            ytests = [y1, y4, y5, y6, y7]

            roc_name = 'roccurve_imbalance_CnnCrispr_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CnnCrispr_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CnnCrispr_model, CnnCrispr_undersampling, CnnCrispr_NearMiss, CnnCrispr_ENN,
                      CnnCrispr_NearMiss_3, CnnCrispr_TomekLinks]

            labels = ['CnnCrispr', 'CnnCrispr_UnderSampling', 'CnnCrispr_NearMiss', 'CnnCrispr_ENN',
                      'CnnCrispr_NearMiss_2', 'CnnCrispr_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CnnCrispr_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CnnCrispr_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CnnCrispr_model, CnnCrispr_SMOTETomek,
                      CnnCrispr_SMOTEENN]

            labels = ['CnnCrispr', 'CnnCrispr_SMOTETomek', 'CnnCrispr_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CnnCrispr_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CnnCrispr_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CnnCrispr': data1, 'CnnCrispr_focal_loss': data2, 'CnnCrispr_GHM': data3,
                         'CnnCrispr_oversampling': data4, 'CnnCrispr_SMOTE': data5, 'CnnCrispr_ADASYN': data6,
                         'CnnCrispr_BorderlineSMOTE': data7,
                         'CnnCrispr_undersampling': data10,
                         'CnnCrispr_NearMiss': data11, 'CnnCrispr_ENN': data12,
                         'CnnCrispr_NearMiss_3': data14, 'CnnCrispr_TomekLinks': data15,
                         'CnnCrispr_SMOTETomek': data16, 'CnnCrispr_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CnnCrispr.csv', index=None)

            data_save = {'test': ytest, 'CnnCrispr': prob1, 'CnnCrispr_focal_loss': prob2, 'CnnCrispr_GHM': prob3,
                         'CnnCrispr_oversampling': prob4, 'CnnCrispr_SMOTE': prob5, 'CnnCrispr_ADASYN': prob6,
                         'CnnCrispr_BorderlineSMOTE': prob7,
                         'CnnCrispr_undersampling': prob10,
                         'CnnCrispr_NearMiss': prob11, 'CnnCrispr_ENN': prob12,
                         'CnnCrispr_NearMiss_3': prob14, 'CnnCrispr_TomekLinks': prob15,
                         'CnnCrispr_SMOTETomek': prob16, 'CnnCrispr_SMOTEENN': prob17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CnnCrispr_prob.csv', index=None)


if flag == 14:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000

            print('dl_crispr SMOTETomek')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SMOTETomek = newnetwork.pre_dl_crispr_SMOTETomek(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('dl_crispr SMOTETomek Test')
            yscore = dl_crispr_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr SMOTEENN')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SMOTEENN = newnetwork.pre_dl_crispr_SMOTEENN(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('dl_crispr SMOTEENN Test')
            yscore = dl_crispr_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


if flag == 13:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000
            print('dl_crispr')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x1, y1, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            dl_crispr_model = newnetwork.pre_dl_crispr(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('dl_crispr Test')
            yscore = dl_crispr_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            prob1 = yscore
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr SMOTE')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x5, y5, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SMOTE = newnetwork.pre_dl_crispr_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('dl_crispr SMOTE Test')
            yscore = dl_crispr_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            prob5 = yscore
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr ADASYN')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x6, y6, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_ADASYN = newnetwork.pre_dl_crispr_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size, epochs, callbacks,
                                                           open_name, retrain)

            print('dl_crispr ADASYN Test')
            yscore = dl_crispr_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            prob6 = yscore
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x7, y7, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_BorderlineSMOTE = newnetwork.pre_dl_crispr_BorderlineSMOTE(test_ds, xval, yval,
                                                                             resampled_steps_per_epoch,
                                                                             resampled_ds, xtrain, ytrain,
                                                                             inputshape, num_classes, batch_size,
                                                                             epochs, callbacks,
                                                                             open_name, retrain)

            print('dl_crispr BorderlineSMOTE Test')
            yscore = dl_crispr_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            prob7 = yscore
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net KMeansSMOTE')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = KMeansSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x8, y8, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                            coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_KMeansSMOTE = newnetwork.pre_CRISPR_Net_KMeansSMOTE(test_ds, xval, yval,
            #                                                                        resampled_steps_per_epoch,
            #                                                                        resampled_ds, xtrain, ytrain,
            #                                                                        inputshape, num_classes, batch_size,
            #                                                                        epochs, callbacks,
            #                                                                        open_name, retrain)
            #
            # print('CRISPR_Net KMeansSMOTE Test')
            # yscore = CRISPR_Net_KMeansSMOTE.predict(x8)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data8 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y8, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr SVMSMOTE')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SVMSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x9, y9, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SVMSMOTE = newnetwork.pre_dl_crispr_SVMSMOTE(test_ds, xval, yval,
                                                               resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size,
                                                               epochs, callbacks,
                                                               open_name, retrain)

            print('dl_crispr SVMSMOTE Test')
            yscore = dl_crispr_SVMSMOTE.predict(x9)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data9 = ypred
            yscore = yscore[:, 1]
            prob9 = yscore
            # print(yscore)
            ytest = np.argmax(y9, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr oversampling')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x4, y4, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_oversampling = newnetwork.pre_dl_crispr_oversampling(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('dl_crispr oversampling Test')
            yscore = dl_crispr_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            prob4 = yscore
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr_focal_loss')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x2, y2, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_focal_loss = newnetwork.pre_dl_crispr_focal_loss(x2, y2, test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size, epochs,
                                                                   callbacks,
                                                                   open_name, retrain)

            print('dl_crispr_focal_loss Test')
            yscore = dl_crispr_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            prob2 = yscore
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr_GHM')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x3, y3, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                    coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_GHM = newnetwork.pre_dl_crispr_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                     resampled_ds, xtrain, ytrain,
                                                     inputshape, num_classes, batch_size, epochs,
                                                     callbacks,
                                                     open_name, retrain)

            print('dl_crispr_GHM Test')
            yscore = dl_crispr_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            prob3 = yscore
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr undersampling')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x10, y10, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_undersampling = newnetwork.pre_dl_crispr_undersampling(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('dl_crispr undersampling Test')
            yscore = dl_crispr_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            prob10 = yscore
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr NearMiss')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x11, y11, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_NearMiss = newnetwork.pre_dl_crispr_NearMiss(test_ds, xval, yval,
                                                               resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size,
                                                               epochs, callbacks,
                                                               open_name, retrain)

            print('dl_crispr NearMiss Test')
            yscore = dl_crispr_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            prob11 = yscore
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net NearMiss_2')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = NearMiss(version=2)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x13, y13, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                              coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_NearMiss_2 = newnetwork.pre_CRISPR_Net_NearMiss_2(test_ds, xval, yval,
            #                                                          resampled_steps_per_epoch,
            #                                                          resampled_ds, xtrain, ytrain,
            #                                                          inputshape, num_classes, batch_size,
            #                                                          epochs, callbacks,
            #                                                          open_name, retrain)
            #
            # print('CRISPR_Net NearMiss_2 Test')
            # yscore = CRISPR_Net_NearMiss_2.predict(x13)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data13 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y13, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr NearMiss_3')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x14, y14, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_NearMiss_3 = newnetwork.pre_dl_crispr_NearMiss_3(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('dl_crispr NearMiss_3 Test')
            yscore = dl_crispr_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            prob14 = yscore
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr TomekLinks')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x15, y15, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_TomekLinks = newnetwork.pre_dl_crispr_TomekLinks(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('dl_crispr TomekLinks Test')
            yscore = dl_crispr_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            prob15 = yscore
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr ENN')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x12, y12, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_ENN = newnetwork.pre_dl_crispr_ENN(test_ds, xval, yval,
                                                     resampled_steps_per_epoch,
                                                     resampled_ds, xtrain, ytrain,
                                                     inputshape, num_classes, batch_size,
                                                     epochs, callbacks,
                                                     open_name, retrain)

            print('dl_crispr ENN Test')
            yscore = dl_crispr_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            prob12 = yscore
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr SMOTETomek')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SMOTETomek = newnetwork.pre_dl_crispr_SMOTETomek(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('dl_crispr SMOTETomek Test')
            yscore = dl_crispr_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('dl_crispr SMOTEENN')
            # type = '14x23'
            open_name = 'encoded20x20' + dataset + 'withoutTsai.pkl'
            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                      coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            dl_crispr_SMOTEENN = newnetwork.pre_dl_crispr_SMOTEENN(test_ds, xval, yval,
                                                               resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size,
                                                               epochs, callbacks,
                                                               open_name, retrain)

            print('dl_crispr SMOTEENN Test')
            yscore = dl_crispr_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [dl_crispr_model, dl_crispr_focal_loss, dl_crispr_GHM]

            labels = ['DL_CRISPR', 'DL_CRISPR_focal_loss', 'DL_CRISPR_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_dl_crispr_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_dl_crispr_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [dl_crispr_model, dl_crispr_oversampling,
                      dl_crispr_SMOTE, dl_crispr_ADASYN, dl_crispr_BorderlineSMOTE,
                      dl_crispr_SVMSMOTE]

            labels = ['DL_CRISPR', 'DL_CRISPR_OverSampling',
                      'DL_CRISPR_SMOTE', 'DL_CRISPR_ADASYN', 'DL_CRISPR_KMeansSMOTE',
                      'DL_CRISPR_SVMSMOTE']

            xtests = [x1, x4, x5, x6, x7, x9]

            ytests = [y1, y4, y5, y6, y7, y9]

            roc_name = 'roccurve_imbalance_dl_crispr_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_dl_crispr_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [dl_crispr_model, dl_crispr_undersampling, dl_crispr_NearMiss, dl_crispr_ENN,
                      dl_crispr_NearMiss_3, dl_crispr_TomekLinks]

            labels = ['DL_CRISPR', 'DL_CRISPR_UnderSampling', 'DL_CRISPR_NearMiss', 'DL_CRISPR_ENN',
                      'DL_CRISPR_NearMiss_2', 'DL_CRISPR_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_dl_crispr_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_dl_crispr_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [dl_crispr_model, dl_crispr_SMOTETomek,
                      dl_crispr_SMOTEENN]

            labels = ['DL_CRISPR', 'DL_CRISPR_SMOTETomek', 'DL_CRISPR_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_dl_crispr_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_dl_crispr_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'dl_crispr': data1, 'dl_crispr_focal_loss': data2, 'dl_crispr_GHM': data3,
                         'dl_crispr_oversampling': data4, 'dl_crispr_SMOTE': data5, 'dl_crispr_ADASYN': data6,
                         'dl_crispr_BorderlineSMOTE': data7,
                         'dl_crispr_SVMSMOTE': data9, 'dl_crispr_undersampling': data10,
                         'dl_crispr_NearMiss': data11, 'dl_crispr_ENN': data12,
                         'dl_crispr_NearMiss_3': data14, 'dl_crispr_TomekLinks': data15,
                         'dl_crispr_SMOTETomek': data16, 'dl_crispr_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_dl_crispr.csv', index=None)

            data_save = {'test': ytest, 'dl_crispr': prob1, 'dl_crispr_focal_loss': prob2, 'dl_crispr_GHM': prob3,
                         'dl_crispr_oversampling': prob4, 'dl_crispr_SMOTE': prob5, 'dl_crispr_ADASYN': prob6,
                         'dl_crispr_BorderlineSMOTE': prob7,
                         'dl_crispr_SVMSMOTE': prob9, 'dl_crispr_undersampling': prob10,
                         'dl_crispr_NearMiss': prob11, 'dl_crispr_ENN': prob12,
                         'dl_crispr_NearMiss_3': prob14, 'dl_crispr_TomekLinks': prob15,
                         'dl_crispr_SMOTETomek': prob16, 'dl_crispr_SMOTEENN': prob17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_dl_crispr_prob.csv', index=None)


if flag == 12:
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000

            encoder_shape = (20, 20)
            seg_len, coding_dim = encoder_shape

            open_name1 = 'encoded20x20' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x, y, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            open_name = 'encoded20x20' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            custom_objects = get_custom_objects()
            my_objects = {'PositionalEncoding': PositionalEncoding}
            custom_objects.update(my_objects)
            CrisprDNT_model = load_model('{}+dl_crispr.h5'.format(open_name),
                               custom_objects=custom_objects)

            yscore = CrisprDNT_model.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data1 = ypred
            prob1 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            custom_objects = get_custom_objects()
            my_objects = {'PositionalEncoding': PositionalEncoding, 'Focal_loss': Focal_loss}
            custom_objects.update(my_objects)
            CrisprDNT_focal_loss = load_model('{}+dl_crispr_focal_loss.h5'.format(open_name),
                               custom_objects=custom_objects)

            yscore = CrisprDNT_focal_loss.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data2 = ypred
            prob2 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            custom_objects = get_custom_objects()
            my_objects = {'PositionalEncoding': PositionalEncoding, 'GHM_Loss': GHM_Loss}
            custom_objects.update(my_objects)
            CrisprDNT_GHM = load_model('{}+dl_crispr_GHM.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_GHM.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data3 = ypred
            prob3 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            custom_objects = get_custom_objects()
            my_objects = {'PositionalEncoding': PositionalEncoding}
            custom_objects.update(my_objects)
            CrisprDNT_oversampling = load_model('{}+dl_crispr_oversampling.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_oversampling.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data4 = ypred
            prob4 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_SMOTE = load_model('{}+dl_crispr_SMOTE.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_SMOTE.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data5 = ypred
            prob5 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_ADASYN = load_model('{}+dl_crispr_ADASYN.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_ADASYN.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data6 = ypred
            prob6 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_BorderlineSMOTE = load_model('{}+dl_crispr_BorderlineSMOTE.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_BorderlineSMOTE.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data7 = ypred
            prob7 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_SVMSMOTE = load_model('{}+dl_crispr_SVMSMOTE.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_SVMSMOTE.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data9 = ypred
            prob9 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_undersampling = load_model('{}+dl_crispr_undersampling.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_undersampling.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data10 = ypred
            prob10 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_NearMiss = load_model('{}+dl_crispr_NearMiss.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_NearMiss.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data11 = ypred
            prob11 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_ENN = load_model('{}+dl_crispr_ENN.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_ENN.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data12 = ypred
            prob12 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_NearMiss_3 = load_model('{}+dl_crispr_NearMiss_3.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_NearMiss_3.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data14 = ypred
            prob14 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_TomekLinks = load_model('{}+dl_crispr_TomekLinks.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_TomekLinks.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data15 = ypred
            prob15 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_SMOTETomek = load_model('{}+dl_crispr_SMOTETomek.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_SMOTETomek.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data16 = ypred
            prob16 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            CrisprDNT_SMOTEENN = load_model('{}+dl_crispr_SMOTEENN.h5'.format(open_name),
                                               custom_objects=custom_objects)

            yscore = CrisprDNT_SMOTEENN.predict(x)
            ypred = np.argmax(yscore, axis=1)
            data17 = ypred
            prob17 = yscore[:, 1]
            ytest = np.argmax(y, axis=1)

            models = [CrisprDNT_model, CrisprDNT_focal_loss, CrisprDNT_GHM]

            labels = ['DL_CRISPR', 'DL_CRISPR_focal_loss', 'DL_CRISPR_GHM']

            xtests = [x, x, x]

            ytests = [y, y, y]

            roc_name = 'roccurve_imbalance_CrisprDNT_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_oversampling,
                      CrisprDNT_SMOTE, CrisprDNT_ADASYN, CrisprDNT_BorderlineSMOTE,
                      CrisprDNT_SVMSMOTE]

            labels = ['DL_CRISPR', 'DL_CRISPR_OverSampling',
                      'DL_CRISPR_SMOTE', 'DL_CRISPR_ADASYN', 'DL_CRISPR_KMeansSMOTE',
                      'DL_CRISPR_SVMSMOTE']

            xtests = [x, x, x, x, x, x]

            ytests = [y, y, y, y, y, y]

            roc_name = 'roccurve_imbalance_CrisprDNT_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_undersampling, CrisprDNT_NearMiss, CrisprDNT_ENN,
                      CrisprDNT_NearMiss_3, CrisprDNT_TomekLinks]

            labels = ['DL_CRISPR', 'DL_CRISPR_UnderSampling', 'DL_CRISPR_NearMiss', 'DL_CRISPR_ENN',
                      'DL_CRISPR_NearMiss_2', 'DL_CRISPR_TomekLinks']

            xtests = [x, x, x, x, x, x]

            ytests = [y, y, y, y, y, y]

            roc_name = 'roccurve_imbalance_CrisprDNT_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_SMOTETomek,
                      CrisprDNT_SMOTEENN]

            labels = ['DL_CRISPR', 'DL_CRISPR_SMOTETomek', 'DL_CRISPR_SMOTEENN']

            xtests = [x, x, x]

            ytests = [y, y, y]

            roc_name = 'roccurve_imbalance_CrisprDNT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'DL_CRISPR': data1, 'DL_CRISPR_focal_loss': data2, 'DL_CRISPR_GHM': data3,
                         'DL_CRISPR_oversampling': data4, 'DL_CRISPR_SMOTE': data5, 'DL_CRISPR_ADASYN': data6,
                         'DL_CRISPR_BorderlineSMOTE': data7,
                         'DL_CRISPR_SVMSMOTE': data9, 'DL_CRISPR_undersampling': data10,
                         'DL_CRISPR_NearMiss': data11, 'DL_CRISPR_ENN': data12,
                         'DL_CRISPR_NearMiss_3': data14, 'DL_CRISPR_TomekLinks': data15,
                         'DL_CRISPR_SMOTETomek': data16, 'DL_CRISPR_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_DL_CRISPR.csv', index=None)

            data_save = {'test': ytest, 'DL_CRISPR': prob1, 'DL_CRISPR_focal_loss': prob2, 'DL_CRISPR_GHM': prob3,
                         'DL_CRISPR_oversampling': prob4, 'DL_CRISPR_SMOTE': prob5, 'DL_CRISPR_ADASYN': prob6,
                         'DL_CRISPR_BorderlineSMOTE': prob7,
                         'DL_CRISPR_SVMSMOTE': prob9, 'DL_CRISPR_undersampling': prob10,
                         'DL_CRISPR_NearMiss': prob11, 'DL_CRISPR_ENN': prob12,
                         'DL_CRISPR_NearMiss_3': prob14, 'DL_CRISPR_TomekLinks': prob15,
                         'DL_CRISPR_SMOTETomek': prob16, 'DL_CRISPR_SMOTEENN': prob17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_DL_CRISPR_prob.csv', index=None)


if flag == 10:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000
            print('CNN_std')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x1, y1, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CNN_std_model = newnetwork.pre_cnn_std(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('CNN_std Test')
            yscore = CNN_std_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            prob1 = yscore
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std SMOTE')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x5, y5, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_SMOTE = newnetwork.pre_cnn_std_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('CNN_std SMOTE Test')
            yscore = CNN_std_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            prob5 = yscore
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std ADASYN')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x6, y6, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_ADASYN = newnetwork.pre_cnn_std_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CNN_std ADASYN Test')
            yscore = CNN_std_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            prob6 = yscore
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x7, y7, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_BorderlineSMOTE = newnetwork.pre_cnn_std_BorderlineSMOTE(test_ds, xval, yval,
                                                                                 resampled_steps_per_epoch,
                                                                                 resampled_ds, xtrain, ytrain,
                                                                                 inputshape, num_classes, batch_size,
                                                                                 epochs, callbacks,
                                                                                 open_name, retrain)

            print('CNN_std BorderlineSMOTE Test')
            yscore = CNN_std_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            prob7 = yscore
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net KMeansSMOTE')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = KMeansSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x8, y8, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                            coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_KMeansSMOTE = newnetwork.pre_CRISPR_Net_KMeansSMOTE(test_ds, xval, yval,
            #                                                                        resampled_steps_per_epoch,
            #                                                                        resampled_ds, xtrain, ytrain,
            #                                                                        inputshape, num_classes, batch_size,
            #                                                                        epochs, callbacks,
            #                                                                        open_name, retrain)
            #
            # print('CRISPR_Net KMeansSMOTE Test')
            # yscore = CRISPR_Net_KMeansSMOTE.predict(x8)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data8 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y8, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std SVMSMOTE')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SVMSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x9, y9, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_SVMSMOTE = newnetwork.pre_cnn_std_SVMSMOTE(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CNN_std SVMSMOTE Test')
            yscore = CNN_std_SVMSMOTE.predict(x9)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data9 = ypred
            yscore = yscore[:, 1]
            prob9 = yscore
            # print(yscore)
            ytest = np.argmax(y9, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std oversampling')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x4, y4, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_oversampling = newnetwork.pre_cnn_std_oversampling(test_ds, xval, yval,
                                                                           resampled_steps_per_epoch,
                                                                           resampled_ds, xtrain, ytrain,
                                                                           inputshape, num_classes, batch_size,
                                                                           epochs, callbacks,
                                                                           open_name, retrain)

            print('CNN_std oversampling Test')
            yscore = CNN_std_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            prob4 = yscore
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std_focal_loss')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x2, y2, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CNN_std_focal_loss = newnetwork.pre_CNN_std_focal_loss(x2, y2, test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size, epochs,
                                                                       callbacks,
                                                                       open_name, retrain)

            print('CNN_std_focal_loss Test')
            yscore = CNN_std_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            prob2 = yscore
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std_GHM')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x3, y3, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CNN_std_GHM = newnetwork.pre_CNN_std_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         inputshape, num_classes, batch_size, epochs,
                                                         callbacks,
                                                         open_name, retrain)

            print('CNN_std_GHM Test')
            yscore = CNN_std_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            prob3 = yscore
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std undersampling')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x10, y10, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_undersampling = newnetwork.pre_cnn_std_undersampling(test_ds, xval, yval,
                                                                             resampled_steps_per_epoch,
                                                                             resampled_ds, xtrain, ytrain,
                                                                             inputshape, num_classes, batch_size,
                                                                             epochs, callbacks,
                                                                             open_name, retrain)

            print('CNN_std undersampling Test')
            yscore = CNN_std_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            prob10 = yscore
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std NearMiss')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x11, y11, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_NearMiss = newnetwork.pre_cnn_std_NearMiss(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CNN_std NearMiss Test')
            yscore = CNN_std_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            prob11 = yscore
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net NearMiss_2')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = NearMiss(version=2)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x13, y13, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                              coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_NearMiss_2 = newnetwork.pre_CRISPR_Net_NearMiss_2(test_ds, xval, yval,
            #                                                          resampled_steps_per_epoch,
            #                                                          resampled_ds, xtrain, ytrain,
            #                                                          inputshape, num_classes, batch_size,
            #                                                          epochs, callbacks,
            #                                                          open_name, retrain)
            #
            # print('CRISPR_Net NearMiss_2 Test')
            # yscore = CRISPR_Net_NearMiss_2.predict(x13)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data13 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y13, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std NearMiss_3')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x14, y14, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_NearMiss_3 = newnetwork.pre_cnn_std_NearMiss_3(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CNN_std NearMiss_3 Test')
            yscore = CNN_std_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            prob14 = yscore
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std TomekLinks')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x15, y15, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_TomekLinks = newnetwork.pre_cnn_std_TomekLinks(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CNN_std TomekLinks Test')
            yscore = CNN_std_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            prob15 = yscore
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std ENN')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x12, y12, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_ENN = newnetwork.pre_cnn_std_ENN(test_ds, xval, yval,
                                                         resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         inputshape, num_classes, batch_size,
                                                         epochs, callbacks,
                                                         open_name, retrain)

            print('CNN_std ENN Test')
            yscore = CNN_std_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            prob12 = yscore
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std SMOTETomek')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_SMOTETomek = newnetwork.pre_cnn_std_SMOTETomek(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CNN_std SMOTETomek Test')
            yscore = CNN_std_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CNN_std SMOTEENN')
            # type = '14x23'
            open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 4)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded4x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_cnn_std_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_cnn_std_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded4x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CNN_std_SMOTEENN = newnetwork.pre_cnn_std_SMOTEENN(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CNN_std SMOTEENN Test')
            yscore = CNN_std_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [CNN_std_model, CNN_std_focal_loss, CNN_std_GHM]

            labels = ['CNN_std', 'CNN_std_focal_loss', 'CNN_std_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CNN_std_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CNN_std_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CNN_std_model, CNN_std_oversampling,
                      CNN_std_SMOTE, CNN_std_ADASYN, CNN_std_BorderlineSMOTE,
                      CNN_std_SVMSMOTE]

            labels = ['CNN_std', 'CNN_std_OverSampling',
                      'CNN_std_SMOTE', 'CNN_std_ADASYN', 'CNN_std_KMeansSMOTE',
                      'CNN_std_SVMSMOTE']

            xtests = [x1, x4, x5, x6, x7, x9]

            ytests = [y1, y4, y5, y6, y7, y9]

            roc_name = 'roccurve_imbalance_CNN_std_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CNN_std_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CNN_std_model, CNN_std_undersampling, CNN_std_NearMiss, CNN_std_ENN,
                      CNN_std_NearMiss_3, CNN_std_TomekLinks]

            labels = ['CNN_std', 'CNN_std_UnderSampling', 'CNN_std_NearMiss', 'CNN_std_ENN',
                      'CNN_std_NearMiss_2', 'CNN_std_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CNN_std_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CNN_std_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CNN_std_model, CNN_std_SMOTETomek,
                      CNN_std_SMOTEENN]

            labels = ['CNN_std', 'CNN_std_SMOTETomek', 'CNN_std_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CNN_std_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CNN_std_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CNN_std': data1, 'CNN_std_focal_loss': data2, 'CNN_std_GHM': data3,
                         'CNN_std_oversampling': data4, 'CNN_std_SMOTE': data5, 'CNN_std_ADASYN': data6,
                         'CNN_std_BorderlineSMOTE': data7,
                         'CNN_std_SVMSMOTE': data9, 'CNN_std_undersampling': data10,
                         'CNN_std_NearMiss': data11, 'CNN_std_ENN': data12,
                         'CNN_std_NearMiss_3': data14, 'CNN_std_TomekLinks': data15,
                         'CNN_std_SMOTETomek': data16, 'CNN_std_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CNN_std.csv', index=None)

            data_save = {'test': ytest, 'CNN_std': prob1, 'CNN_std_focal_loss': prob2, 'CNN_std_GHM': prob3,
                         'CNN_std_oversampling': prob4, 'CNN_std_SMOTE': prob5, 'CNN_std_ADASYN': prob6,
                         'CNN_std_BorderlineSMOTE': prob7,
                         'CNN_std_SVMSMOTE': prob9, 'CNN_std_undersampling': prob10,
                         'CNN_std_NearMiss': prob11, 'CNN_std_ENN': prob12,
                         'CNN_std_NearMiss_3': prob14, 'CNN_std_TomekLinks': prob15,
                         'CNN_std_SMOTETomek': prob16, 'CNN_std_SMOTEENN': prob17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CNN_std_prob.csv', index=None)

if flag == 8:
    # epochs = 60
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000
            #
            print('CrisprDNT')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x1, y1, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CrisprDNT_model = newnetwork.pre_CrisprDNT_model(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('CrisprDNT Test')
            yscore = CrisprDNT_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            prob1 = yscore
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT SMOTE')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x5, y5, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_SMOTE = newnetwork.pre_CrisprDNT_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                             resampled_ds, xtrain, ytrain,
                                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                                             open_name, retrain)

            print('CrisprDNT SMOTE Test')
            yscore = CrisprDNT_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            prob5 = yscore
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT ADASYN')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x6, y6, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_ADASYN = newnetwork.pre_CrisprDNT_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CrisprDNT ADASYN Test')
            yscore = CrisprDNT_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            prob6 = yscore
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x7, y7, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_BorderlineSMOTE = newnetwork.pre_CrisprDNT_BorderlineSMOTE(test_ds, xval, yval,
                                                                                 resampled_steps_per_epoch,
                                                                                 resampled_ds, xtrain, ytrain,
                                                                                 inputshape, num_classes, batch_size,
                                                                                 epochs, callbacks,
                                                                                 open_name, retrain)

            print('CrisprDNT BorderlineSMOTE Test')
            yscore = CrisprDNT_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            prob7 = yscore
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net KMeansSMOTE')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = KMeansSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x8, y8, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                            coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_KMeansSMOTE = newnetwork.pre_CRISPR_Net_KMeansSMOTE(test_ds, xval, yval,
            #                                                                        resampled_steps_per_epoch,
            #                                                                        resampled_ds, xtrain, ytrain,
            #                                                                        inputshape, num_classes, batch_size,
            #                                                                        epochs, callbacks,
            #                                                                        open_name, retrain)
            #
            # print('CRISPR_Net KMeansSMOTE Test')
            # yscore = CRISPR_Net_KMeansSMOTE.predict(x8)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data8 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y8, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT SVMSMOTE')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SVMSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x9, y9, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_SVMSMOTE = newnetwork.pre_CrisprDNT_SVMSMOTE(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CrisprDNT SVMSMOTE Test')
            yscore = CrisprDNT_SVMSMOTE.predict(x9)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data9 = ypred
            yscore = yscore[:, 1]
            prob9 = yscore
            # print(yscore)
            ytest = np.argmax(y9, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT oversampling')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x4, y4, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_oversampling = newnetwork.pre_CrisprDNT_oversampling(test_ds, xval, yval,
                                                                           resampled_steps_per_epoch,
                                                                           resampled_ds, xtrain, ytrain,
                                                                           inputshape, num_classes, batch_size,
                                                                           epochs, callbacks,
                                                                           open_name, retrain)

            print('CrisprDNT oversampling Test')
            yscore = CrisprDNT_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            prob4 = yscore
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT_focal_loss')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x2, y2, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CrisprDNT_focal_loss = newnetwork.pre_CrisprDNT_focal_loss(x2, y2, test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size, epochs,
                                                                       callbacks,
                                                                       open_name, retrain)

            print('CrisprDNT_focal_loss Test')
            yscore = CrisprDNT_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            prob2 = yscore
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT_GHM')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x3, y3, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                            coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CrisprDNT_GHM = newnetwork.pre_CrisprDNT_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         inputshape, num_classes, batch_size, epochs,
                                                         callbacks,
                                                         open_name, retrain)

            print('CrisprDNT_GHM Test')
            yscore = CrisprDNT_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            prob3 = yscore
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT undersampling')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x10, y10, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_undersampling = newnetwork.pre_CrisprDNT_undersampling(test_ds, xval, yval,
                                                                             resampled_steps_per_epoch,
                                                                             resampled_ds, xtrain, ytrain,
                                                                             inputshape, num_classes, batch_size,
                                                                             epochs, callbacks,
                                                                             open_name, retrain)

            print('CrisprDNT undersampling Test')
            yscore = CrisprDNT_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            prob10 = yscore
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT NearMiss')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x11, y11, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_NearMiss = newnetwork.pre_CrisprDNT_NearMiss(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CrisprDNT NearMiss Test')
            yscore = CrisprDNT_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            prob11 = yscore
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net NearMiss_2')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = NearMiss(version=2)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x13, y13, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                              coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_NearMiss_2 = newnetwork.pre_CRISPR_Net_NearMiss_2(test_ds, xval, yval,
            #                                                          resampled_steps_per_epoch,
            #                                                          resampled_ds, xtrain, ytrain,
            #                                                          inputshape, num_classes, batch_size,
            #                                                          epochs, callbacks,
            #                                                          open_name, retrain)
            #
            # print('CRISPR_Net NearMiss_2 Test')
            # yscore = CRISPR_Net_NearMiss_2.predict(x13)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data13 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y13, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT NearMiss_3')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x14, y14, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_NearMiss_3 = newnetwork.pre_CrisprDNT_NearMiss_3(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CrisprDNT NearMiss_3 Test')
            yscore = CrisprDNT_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            prob14 = yscore
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT TomekLinks')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x15, y15, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_TomekLinks = newnetwork.pre_CrisprDNT_TomekLinks(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CrisprDNT TomekLinks Test')
            yscore = CrisprDNT_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            prob15 = yscore
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT ENN')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x12, y12, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_ENN = newnetwork.pre_CrisprDNT_ENN(test_ds, xval, yval,
                                                         resampled_steps_per_epoch,
                                                         resampled_ds, xtrain, ytrain,
                                                         inputshape, num_classes, batch_size,
                                                         epochs, callbacks,
                                                         open_name, retrain)

            print('CrisprDNT ENN Test')
            yscore = CrisprDNT_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            prob12 = yscore
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT SMOTETomek')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_SMOTETomek = newnetwork.pre_CrisprDNT_SMOTETomek(test_ds, xval, yval,
                                                                       resampled_steps_per_epoch,
                                                                       resampled_ds, xtrain, ytrain,
                                                                       inputshape, num_classes, batch_size,
                                                                       epochs, callbacks,
                                                                       open_name, retrain)

            print('CrisprDNT SMOTETomek Test')
            yscore = CrisprDNT_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            prob16 = yscore
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CrisprDNT SMOTEENN')
            # type = '14x23'
            open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 14)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedmismatchtype14x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                              coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedmismatchtype14x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CrisprDNT_SMOTEENN = newnetwork.pre_CrisprDNT_SMOTEENN(test_ds, xval, yval,
                                                                   resampled_steps_per_epoch,
                                                                   resampled_ds, xtrain, ytrain,
                                                                   inputshape, num_classes, batch_size,
                                                                   epochs, callbacks,
                                                                   open_name, retrain)

            print('CrisprDNT SMOTEENN Test')
            yscore = CrisprDNT_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            prob17 = yscore
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [CrisprDNT_model, CrisprDNT_focal_loss, CrisprDNT_GHM]

            labels = ['CrisprDNT', 'CrisprDNT_focal_loss', 'CrisprDNT_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CrisprDNT_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_oversampling,
                      CrisprDNT_SMOTE, CrisprDNT_ADASYN, CrisprDNT_BorderlineSMOTE,
                      CrisprDNT_SVMSMOTE]

            labels = ['CrisprDNT', 'CrisprDNT_OverSampling',
                      'CrisprDNT_SMOTE', 'CrisprDNT_ADASYN', 'CrisprDNT_KMeansSMOTE',
                      'CrisprDNT_SVMSMOTE']

            xtests = [x1, x4, x5, x6, x7, x9]

            ytests = [y1, y4, y5, y6, y7, y9]

            roc_name = 'roccurve_imbalance_CrisprDNT_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_undersampling, CrisprDNT_NearMiss, CrisprDNT_ENN,
                      CrisprDNT_NearMiss_3, CrisprDNT_TomekLinks]

            labels = ['CrisprDNT', 'CrisprDNT_UnderSampling', 'CrisprDNT_NearMiss', 'CrisprDNT_ENN',
                      'CrisprDNT_NearMiss_2', 'CrisprDNT_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CrisprDNT_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CrisprDNT_model, CrisprDNT_SMOTETomek,
                      CrisprDNT_SMOTEENN]

            labels = ['CrisprDNT', 'CrisprDNT_SMOTETomek', 'CrisprDNT_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CrisprDNT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CrisprDNT_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CrisprDNT': data1, 'CrisprDNT_focal_loss': data2, 'CrisprDNT_GHM': data3,
                         'CrisprDNT_oversampling': data4, 'CrisprDNT_SMOTE': data5, 'CrisprDNT_ADASYN': data6,
                         'CrisprDNT_BorderlineSMOTE': data7,
                         'CrisprDNT_SVMSMOTE': data9, 'CrisprDNT_undersampling': data10,
                         'CrisprDNT_NearMiss': data11, 'CrisprDNT_ENN': data12,
                         'CrisprDNT_NearMiss_3': data14, 'CrisprDNT_TomekLinks': data15,
                         'CrisprDNT_SMOTETomek': data16, 'CrisprDNT_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_CrisprDNT.csv', index=None)

if flag == 7:
    # epochs = 1
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000
            #
            print('CRISPR_IP')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x1, y1, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_IP_model = newnetwork.pre_CRISPR_IP_model(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CRISPR_IP Test')
            yscore = CRISPR_IP_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP SMOTE')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x5, y5, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_SMOTE = newnetwork.pre_CRISPR_IP_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CRISPR_IP SMOTE Test')
            yscore = CRISPR_IP_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP ADASYN')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x6, y6, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_ADASYN = newnetwork.pre_CRISPR_IP_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                                 resampled_ds, xtrain, ytrain,
                                                                 inputshape, num_classes, batch_size, epochs, callbacks,
                                                                 open_name, retrain)

            print('CRISPR_IP ADASYN Test')
            yscore = CRISPR_IP_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x7, y7, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_BorderlineSMOTE = newnetwork.pre_CRISPR_IP_BorderlineSMOTE(test_ds, xval, yval,
                                                                                   resampled_steps_per_epoch,
                                                                                   resampled_ds, xtrain, ytrain,
                                                                                   inputshape, num_classes, batch_size,
                                                                                   epochs, callbacks,
                                                                                   open_name, retrain)

            print('CRISPR_IP BorderlineSMOTE Test')
            yscore = CRISPR_IP_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net KMeansSMOTE')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = KMeansSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x8, y8, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                            coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_KMeansSMOTE = newnetwork.pre_CRISPR_Net_KMeansSMOTE(test_ds, xval, yval,
            #                                                                        resampled_steps_per_epoch,
            #                                                                        resampled_ds, xtrain, ytrain,
            #                                                                        inputshape, num_classes, batch_size,
            #                                                                        epochs, callbacks,
            #                                                                        open_name, retrain)
            #
            # print('CRISPR_Net KMeansSMOTE Test')
            # yscore = CRISPR_Net_KMeansSMOTE.predict(x8)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data8 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y8, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP SVMSMOTE')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SVMSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x9, y9, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_SVMSMOTE = newnetwork.pre_CRISPR_IP_SVMSMOTE(test_ds, xval, yval,
                                                                     resampled_steps_per_epoch,
                                                                     resampled_ds, xtrain, ytrain,
                                                                     inputshape, num_classes, batch_size,
                                                                     epochs, callbacks,
                                                                     open_name, retrain)

            print('CRISPR_IP SVMSMOTE Test')
            yscore = CRISPR_IP_SVMSMOTE.predict(x9)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data9 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y9, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP oversampling')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x4, y4, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_oversampling = newnetwork.pre_CRISPR_IP_oversampling(test_ds, xval, yval,
                                                                             resampled_steps_per_epoch,
                                                                             resampled_ds, xtrain, ytrain,
                                                                             inputshape, num_classes, batch_size,
                                                                             epochs, callbacks,
                                                                             open_name, retrain)

            print('CRISPR_IP oversampling Test')
            yscore = CRISPR_IP_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP_focal_loss')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x2, y2, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_IP_focal_loss = newnetwork.pre_CRISPR_IP_focal_loss(x2, y2, test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size, epochs,
                                                                         callbacks,
                                                                         open_name, retrain)

            print('CRISPR_IP_focal_loss Test')
            yscore = CRISPR_IP_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP_GHM')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x3, y3, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_IP_GHM = newnetwork.pre_CRISPR_IP_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size, epochs,
                                                           callbacks,
                                                           open_name, retrain)

            print('CRISPR_IP_GHM Test')
            yscore = CRISPR_IP_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP undersampling')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x10, y10, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_undersampling = newnetwork.pre_CRISPR_IP_undersampling(test_ds, xval, yval,
                                                                               resampled_steps_per_epoch,
                                                                               resampled_ds, xtrain, ytrain,
                                                                               inputshape, num_classes, batch_size,
                                                                               epochs, callbacks,
                                                                               open_name, retrain)

            print('CRISPR_IP undersampling Test')
            yscore = CRISPR_IP_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP NearMiss')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x11, y11, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_NearMiss = newnetwork.pre_CRISPR_IP_NearMiss(test_ds, xval, yval,
                                                                     resampled_steps_per_epoch,
                                                                     resampled_ds, xtrain, ytrain,
                                                                     inputshape, num_classes, batch_size,
                                                                     epochs, callbacks,
                                                                     open_name, retrain)

            print('CRISPR_IP NearMiss Test')
            yscore = CRISPR_IP_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net NearMiss_2')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = NearMiss(version=2)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x13, y13, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                              coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_NearMiss_2 = newnetwork.pre_CRISPR_Net_NearMiss_2(test_ds, xval, yval,
            #                                                          resampled_steps_per_epoch,
            #                                                          resampled_ds, xtrain, ytrain,
            #                                                          inputshape, num_classes, batch_size,
            #                                                          epochs, callbacks,
            #                                                          open_name, retrain)
            #
            # print('CRISPR_Net NearMiss_2 Test')
            # yscore = CRISPR_Net_NearMiss_2.predict(x13)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data13 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y13, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP NearMiss_3')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x14, y14, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_NearMiss_3 = newnetwork.pre_CRISPR_IP_NearMiss_3(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('CRISPR_IP NearMiss_3 Test')
            yscore = CRISPR_IP_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP TomekLinks')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x15, y15, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_TomekLinks = newnetwork.pre_CRISPR_IP_TomekLinks(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('CRISPR_IP TomekLinks Test')
            yscore = CRISPR_IP_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP ENN')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x12, y12, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_ENN = newnetwork.pre_CRISPR_IP_ENN(test_ds, xval, yval,
                                                           resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size,
                                                           epochs, callbacks,
                                                           open_name, retrain)

            print('CRISPR_IP ENN Test')
            yscore = CRISPR_IP_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP SMOTETomek')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_SMOTETomek = newnetwork.pre_CRISPR_IP_SMOTETomek(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('CRISPR_IP SMOTETomek Test')
            yscore = CRISPR_IP_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_IP SMOTEENN')
            # type = '14x23'
            open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 9)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encodedposition9x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encodedposition9x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_IP_SMOTEENN = newnetwork.pre_CRISPR_IP_SMOTEENN(test_ds, xval, yval,
                                                                     resampled_steps_per_epoch,
                                                                     resampled_ds, xtrain, ytrain,
                                                                     inputshape, num_classes, batch_size,
                                                                     epochs, callbacks,
                                                                     open_name, retrain)

            print('CRISPR_IP SMOTEENN Test')
            yscore = CRISPR_IP_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            models = [CRISPR_IP_model, CRISPR_IP_focal_loss, CRISPR_IP_GHM]

            labels = ['CRISPR_IP', 'CRISPR_IP_focal_loss', 'CRISPR_IP_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CRISPR_IP_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_IP_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_IP_model, CRISPR_IP_oversampling,
                      CRISPR_IP_SMOTE, CRISPR_IP_ADASYN, CRISPR_IP_BorderlineSMOTE,
                      CRISPR_IP_SVMSMOTE]

            labels = ['CRISPR_IP', 'CRISPR_IP_OverSampling',
                      'CRISPR_IP_SMOTE', 'CRISPR_IP_ADASYN', 'CRISPR_IP_KMeansSMOTE',
                      'CRISPR_IP_SVMSMOTE']

            xtests = [x1, x4, x5, x6, x7, x9]

            ytests = [y1, y4, y5, y6, y7, y9]

            roc_name = 'roccurve_imbalance_CRISPR_IP_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_IP_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_IP_model, CRISPR_IP_undersampling, CRISPR_IP_NearMiss, CRISPR_IP_ENN,
                      CRISPR_IP_NearMiss_3, CRISPR_IP_TomekLinks]

            labels = ['CRISPR_IP', 'CRISPR_IP_UnderSampling', 'CRISPR_IP_NearMiss', 'CRISPR_IP_ENN',
                      'CRISPR_IP_NearMiss_2', 'CRISPR_IP_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CRISPR_IP_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_IP_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            models = [CRISPR_IP_model, CRISPR_IP_SMOTETomek,
                      CRISPR_IP_SMOTEENN]

            labels = ['CRISPR_IP', 'CRISPR_IP_SMOTETomek', 'CRISPR_IP_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CRISPR_IP_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_IP_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CRISPR_IP': data1, 'CRISPR_IP_focal_loss': data2, 'CRISPR_IP_GHM': data3,
                         'CRISPR_IP_oversampling': data4, 'CRISPR_IP_SMOTE': data5, 'CRISPR_IP_ADASYN': data6,
                         'CRISPR_IP_BorderlineSMOTE': data7,
                         'CRISPR_IP_SVMSMOTE': data9, 'CRISPR_IP_undersampling': data10,
                         'CRISPR_Ip_NearMiss': data11, 'CRISPR_IP_ENN': data12,
                         'CRISPR_IP_NearMiss_3': data14, 'CRISPR_IP_TomekLinks': data15,
                         'CRISPR_IP_SMOTETomek': data16, 'CRISPR_IP_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '_IP.csv', index=None)

if flag == 6:
    # epochs = 1
    # retrain = False
    for dataset in list_dataset:
        for t_dataset in test_dataset:
            if t_dataset == 'hek293t':
                batch_size = 10000
            else:
                batch_size = 10000
            #
            print('CRISPR_Net')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )


            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target


            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain,xval,yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train,x_val,y_val, seg_len, coding_dim, num_classes)

            x1,y1, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images,loaddata1.target,seg_len, coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5],seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_Net_model = newnetwork.pre_CRISPR_Net_model(test_ds,xval,yval,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size, epochs, callbacks,
                                                           open_name, retrain)

            print('CRISPR_Net Test')
            yscore = CRISPR_Net_model.predict(x1)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data1 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y1, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net SMOTE')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x5, y5, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_SMOTE = newnetwork.pre_CRISPR_Net_SMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CRISPR_Net SMOTE Test')
            yscore = CRISPR_Net_SMOTE.predict(x5)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data5 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y5, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net ADASYN')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = ADASYN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x6, y6, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_ADASYN = newnetwork.pre_CRISPR_Net_ADASYN(test_ds, xval, yval, resampled_steps_per_epoch,
                                                               resampled_ds, xtrain, ytrain,
                                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                                               open_name, retrain)

            print('CRISPR_Net ADASYN Test')
            yscore = CRISPR_Net_ADASYN.predict(x6)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data6 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y6, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net BorderlineSMOTE')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = BorderlineSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x7, y7, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_BorderlineSMOTE = newnetwork.pre_CRISPR_Net_BorderlineSMOTE(test_ds, xval, yval, resampled_steps_per_epoch,
                                                                 resampled_ds, xtrain, ytrain,
                                                                 inputshape, num_classes, batch_size, epochs, callbacks,
                                                                 open_name, retrain)

            print('CRISPR_Net BorderlineSMOTE Test')
            yscore = CRISPR_Net_BorderlineSMOTE.predict(x7)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data7 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y7, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net KMeansSMOTE')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = KMeansSMOTE(random_state=40)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x8, y8, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                            coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_KMeansSMOTE = newnetwork.pre_CRISPR_Net_KMeansSMOTE(test_ds, xval, yval,
            #                                                                        resampled_steps_per_epoch,
            #                                                                        resampled_ds, xtrain, ytrain,
            #                                                                        inputshape, num_classes, batch_size,
            #                                                                        epochs, callbacks,
            #                                                                        open_name, retrain)
            #
            # print('CRISPR_Net KMeansSMOTE Test')
            # yscore = CRISPR_Net_KMeansSMOTE.predict(x8)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data8 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y8, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net SVMSMOTE')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SVMSMOTE(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x9, y9, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_SVMSMOTE = newnetwork.pre_CRISPR_Net_SVMSMOTE(test_ds, xval, yval,
                                                                           resampled_steps_per_epoch,
                                                                           resampled_ds, xtrain, ytrain,
                                                                           inputshape, num_classes, batch_size,
                                                                           epochs, callbacks,
                                                                           open_name, retrain)

            print('CRISPR_Net SVMSMOTE Test')
            yscore = CRISPR_Net_SVMSMOTE.predict(x9)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data9 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y9, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net oversampling')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomOverSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x4, y4, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_oversampling = newnetwork.pre_CRISPR_Net_oversampling(test_ds, xval, yval, resampled_steps_per_epoch,
                                                                 resampled_ds, xtrain, ytrain,
                                                                 inputshape, num_classes, batch_size, epochs, callbacks,
                                                                 open_name, retrain)

            print('CRISPR_net oversampling Test')
            yscore = CRISPR_Net_oversampling.predict(x4)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data4 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y4, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


            print('CRISPR_Net_focal_loss')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x2, y2, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_Net_focal_loss = newnetwork.pre_CRISPR_Net_focal_loss(x2, y2, test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size, epochs,
                                                                         callbacks,
                                                                         open_name, retrain)

            print('CRISPR_Net_focal_loss Test')
            yscore = CRISPR_Net_focal_loss.predict(x2)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data2 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y2, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net_GHM')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x3, y3, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = False

            CRISPR_Net_GHM = newnetwork.pre_CRISPR_Net_GHM(x3, y3, test_ds, xval, yval, resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size, epochs,
                                                           callbacks,
                                                           open_name, retrain)

            print('CRISPR_Net_GHM Test')
            yscore = CRISPR_Net_GHM.predict(x3)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data3 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y3, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net undersampling')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = RandomUnderSampler(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x10, y10, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_undersampling = newnetwork.pre_CRISPR_Net_undersampling(test_ds, xval, yval, resampled_steps_per_epoch,
                                                                 resampled_ds, xtrain, ytrain,
                                                                 inputshape, num_classes, batch_size, epochs, callbacks,
                                                                 open_name, retrain)

            print('CRISPR_Net undersampling Test')
            yscore = CRISPR_Net_undersampling.predict(x10)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data10 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y10, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net NearMiss')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x11, y11, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_NearMiss = newnetwork.pre_CRISPR_Net_NearMiss(test_ds, xval, yval,
                                                                               resampled_steps_per_epoch,
                                                                               resampled_ds, xtrain, ytrain,
                                                                               inputshape, num_classes, batch_size,
                                                                               epochs, callbacks,
                                                                               open_name, retrain)

            print('CRISPR_Net NearMiss Test')
            yscore = CRISPR_Net_NearMiss.predict(x11)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data11 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y11, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            # print('CRISPR_Net NearMiss_2')
            # # type = '14x23'
            # open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            # encoder_shape = (23, 6)
            # seg_len, coding_dim = encoder_shape
            #
            # print('load data!')
            # print('load data!')
            # print(open_name)
            #
            # loaddata = pkl.load(
            #     open(flpath + open_name, 'rb'),
            #     encoding='latin1'
            # )
            #
            # open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            # loaddata1 = pkl.load(
            #     open(flpath + open_name1, 'rb'),
            #     encoding='latin1'
            # )
            #
            # x_train, x_val, y_train, y_val = train_test_split(
            #     loaddata.images,
            #     loaddata.target,  # loaddata.target,
            #     stratify=pd.Series(loaddata.target),
            #     test_size=0.2,
            #     shuffle=True,
            #     random_state=42)
            #
            # x_train, y_train = loaddata.images, loaddata.target
            #
            # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # # print(x_train)
            # oversample = NearMiss(version=2)
            # x_train, y_train = oversample.fit_resample(x_train, y_train)
            #
            # neg = 0
            # for i in y_train:
            #     if i == 0:
            #         neg += 1
            # print(neg)
            #
            # xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
            #     x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)
            #
            # x13, y13, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
            #                                                              coding_dim, num_classes)
            #
            # #
            # # print(ytrain)
            # # print(guideseq_y)
            #
            # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # # print(train_ds)
            # pos_indices = y_train == 1
            # # print(pos_indices)
            # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # # print(pos_y)
            # # print(1)
            # # print(pos_y)
            # print(len(pos_y))
            # print(len(neg_y))
            #
            # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # # print(pos_ds)
            # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
            #
            # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # # print(resampled_ds)
            # # for features, labels in resampled_ds:
            # #     print(labels)
            # resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            # print(resampled_steps_per_epoch)
            #
            # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            # test_ds = test_ds.batch(batch_size)
            #
            # print('Training!!')
            #
            # open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'
            #
            # retrain = True
            #
            # CRISPR_Net_NearMiss_2 = newnetwork.pre_CRISPR_Net_NearMiss_2(test_ds, xval, yval,
            #                                                          resampled_steps_per_epoch,
            #                                                          resampled_ds, xtrain, ytrain,
            #                                                          inputshape, num_classes, batch_size,
            #                                                          epochs, callbacks,
            #                                                          open_name, retrain)
            #
            # print('CRISPR_Net NearMiss_2 Test')
            # yscore = CRISPR_Net_NearMiss_2.predict(x13)
            # # print(yscore)
            # ypred = np.argmax(yscore, axis=1)
            # print(ypred)
            # data13 = ypred
            # yscore = yscore[:, 1]
            # # print(yscore)
            # ytest = np.argmax(y13, axis=1)
            # # print(ytest)
            # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
            #              average_precision_score]
            # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            # eval_fun_types = [True, True, True, True, False, False]
            # for index_f, function in enumerate(eval_funs):
            #     if eval_fun_types[index_f]:
            #         score = np.round(function(ytest, ypred), 4)
            #     else:
            #         score = np.round(function(ytest, yscore), 4)
            #         if index_f == 4:
            #             roc_auc = score
            #         if index_f == 5:
            #             precision, recall, thresholds = precision_recall_curve(ytest, yscore)
            #             score = np.round(auc(recall, precision), 4)
            #             pr_auc = score
            #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net NearMiss_3')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = NearMiss(version=3)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x14, y14, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_NearMiss_3 = newnetwork.pre_CRISPR_Net_NearMiss_3(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('CRISPR_Net NearMiss_3 Test')
            yscore = CRISPR_Net_NearMiss_3.predict(x14)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data14 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y14, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net TomekLinks')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = TomekLinks()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x15, y15, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_TomekLinks = newnetwork.pre_CRISPR_Net_TomekLinks(test_ds, xval, yval,
                                                                         resampled_steps_per_epoch,
                                                                         resampled_ds, xtrain, ytrain,
                                                                         inputshape, num_classes, batch_size,
                                                                         epochs, callbacks,
                                                                         open_name, retrain)

            print('CRISPR_Net TomekLinks Test')
            yscore = CRISPR_Net_TomekLinks.predict(x15)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data15 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y15, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net ENN')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = EditedNearestNeighbours()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x12, y12, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                       coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_ENN = newnetwork.pre_CRISPR_Net_ENN(test_ds, xval, yval,
                                                                     resampled_steps_per_epoch,
                                                                     resampled_ds, xtrain, ytrain,
                                                                     inputshape, num_classes, batch_size,
                                                                     epochs, callbacks,
                                                                     open_name, retrain)

            print('CRISPR_Net ENN Test')
            yscore = CRISPR_Net_ENN.predict(x12)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data12 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y12, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net SMOTETomek')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTETomek(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x16, y16, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_SMOTETomek = newnetwork.pre_CRISPR_Net_SMOTETomek(test_ds, xval, yval,
                                                           resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size,
                                                           epochs, callbacks,
                                                           open_name, retrain)

            print('CRISPR_Net SMOTETomek Test')
            yscore = CRISPR_Net_SMOTETomek.predict(x16)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data16 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y16, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            print('CRISPR_Net SMOTEENN')
            # type = '14x23'
            open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
            encoder_shape = (23, 6)
            seg_len, coding_dim = encoder_shape

            print('load data!')
            print('load data!')
            print(open_name)

            loaddata = pkl.load(
                open(flpath + open_name, 'rb'),
                encoding='latin1'
            )

            open_name1 = 'encoded6x23' + t_dataset + 'withoutTsai.pkl'
            loaddata1 = pkl.load(
                open(flpath + open_name1, 'rb'),
                encoding='latin1'
            )

            x_train, x_val, y_train, y_val = train_test_split(
                loaddata.images,
                loaddata.target,  # loaddata.target,
                stratify=pd.Series(loaddata.target),
                test_size=0.2,
                shuffle=True,
                random_state=42)

            x_train, y_train = loaddata.images, loaddata.target

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            # print(x_train)
            oversample = SMOTEENN(random_state=40)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

            neg = 0
            for i in y_train:
                if i == 0:
                    neg += 1
            print(neg)

            xtrain, ytrain, xval, yval, inputshape = newnetwork.ppre_CRISPR_Net_transformIO(
                x_train, y_train, x_val, y_val, seg_len, coding_dim, num_classes)

            x17, y17, inputshape = newnetwork.pre_CRISPR_Net_transformIO(loaddata1.images, loaddata1.target, seg_len,
                                                                         coding_dim, num_classes)

            #
            # print(ytrain)
            # print(guideseq_y)

            # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            # print(train_ds)
            pos_indices = y_train == 1
            # print(pos_indices)
            pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
            pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
            # print(pos_y)
            # print(1)
            # print(pos_y)
            print(len(pos_y))
            print(len(neg_y))

            pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
            # print(pos_ds)
            neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
            resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
            # print(resampled_ds)
            # for features, labels in resampled_ds:
            #     print(labels)
            resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
            print(resampled_steps_per_epoch)

            test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
            test_ds = test_ds.batch(batch_size)

            print('Training!!')

            open_name = 'encoded6x23' + dataset + '_' + t_dataset + 'withoutTsai.pkl'

            retrain = True

            CRISPR_Net_SMOTEENN = newnetwork.pre_CRISPR_Net_SMOTEENN(test_ds, xval, yval,
                                                           resampled_steps_per_epoch,
                                                           resampled_ds, xtrain, ytrain,
                                                           inputshape, num_classes, batch_size,
                                                           epochs, callbacks,
                                                           open_name, retrain)

            print('CRISPR_Net SMOTEENN Test')
            yscore = CRISPR_Net_SMOTEENN.predict(x17)
            # print(yscore)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            data17 = ypred
            yscore = yscore[:, 1]
            # print(yscore)
            ytest = np.argmax(y17, axis=1)
            # print(ytest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                         average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest, ypred), 4)
                else:
                    score = np.round(function(ytest, yscore), 4)
                    if index_f == 4:
                        roc_auc = score
                    if index_f == 5:
                        precision, recall, thresholds = precision_recall_curve(ytest, yscore)
                        score = np.round(auc(recall, precision), 4)
                        pr_auc = score
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


            models = [CRISPR_Net_model, CRISPR_Net_focal_loss, CRISPR_Net_GHM]

            labels = ['CRISPR_Net', 'CRISPR_Net_focal_loss', 'CRISPR_Net_GHM']

            xtests = [x1, x2, x3]

            ytests = [y1, y2, y3]

            roc_name = 'roccurve_imbalance_CRISPR_Net_loss' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_Net_loss' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)


            models = [CRISPR_Net_model, CRISPR_Net_oversampling,
                      CRISPR_Net_SMOTE, CRISPR_Net_ADASYN, CRISPR_Net_BorderlineSMOTE,
                      CRISPR_Net_SVMSMOTE]

            labels = ['CRISPR_Net', 'CRISPR_Net_OverSampling',
                      'CRISPR_Net_SMOTE', 'CRISPR_Net_ADASYN', 'CRISPR_Net_KMeansSMOTE',
                      'CRISPR_Net_SVMSMOTE']

            xtests = [x1, x4, x5, x6, x7, x9]

            ytests = [y1, y4, y5, y6, y7, y9]

            roc_name = 'roccurve_imbalance_CRISPR_Net_oversampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_Net_oversampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)



            models = [CRISPR_Net_model, CRISPR_Net_undersampling, CRISPR_Net_NearMiss, CRISPR_Net_ENN,
                      CRISPR_Net_NearMiss_3, CRISPR_Net_TomekLinks]

            labels = ['CRISPR_Net', 'CRISPR_Net_UnderSampling', 'CRISPR_Net_NearMiss', 'CRISPR_Net_ENN',
                      'CRISPR_Net_NearMiss_2',  'CRISPR_Net_TomekLinks']

            xtests = [x1, x10, x11, x12, x14, x15]

            ytests = [y1, y10, y11, y12, y14, y15]

            roc_name = 'roccurve_imbalance_CRISPR_Net_undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_Net_undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)



            models = [CRISPR_Net_model, CRISPR_Net_SMOTETomek,
                      CRISPR_Net_SMOTEENN]

            labels = ['CRISPR_Net', 'CRISPR_Net_SMOTETomek', 'CRISPR_Net_SMOTEENN']

            xtests = [x1, x16, x17]

            ytests = [y1, y16, y17]

            roc_name = 'roccurve_imbalance_CRISPR_Net_over&undersampling' + dataset + '_' + t_dataset + '.pdf'
            pr_name = 'precisionrecallcurve_imbalance_CRISPR_Net_over&undersampling' + dataset + '_' + t_dataset + '.pdf'

            newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

            newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

            data_save = {'test': ytest, 'CRISPR_Net': data1, 'CRISPR_Net_focal_loss': data2, 'CRISPR_Net_GHM': data3,
                         'CRISPR_Net_oversampling': data4, 'CRISPR_Net_SMOTE': data5, 'CRISPR_Net_ADASYN': data6,
                         'CRISPR_Net_BorderlineSMOTE': data7,
                         'CRISPR_Net_SVMSMOTE': data9, 'CRISPR_Net_undersampling': data10,
                         'CRISPR_Net_NearMiss': data11, 'CRISPR_Net_ENN': data12,
                         'CRISPR_Net_NearMiss_3': data14, 'CRISPR_Net_TomekLinks': data15,
                         'CRISPR_Net_SMOTETomek': data16, 'CRISPR_Net_SMOTEENN': data17}

            data_save = pd.DataFrame(data_save)

            data_save.to_csv(dataset + '_' + t_dataset + '.csv', index=None)


print('End of the training!!')