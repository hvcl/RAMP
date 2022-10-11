import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import logging
import argparse
from datetime import datetime
import random
from copy import deepcopy

from scipy.io import loadmat
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score, precision_recall_curve, auc

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_metrics(label, tot_logistic, args, out_paths):
    n_cells, n_drugs = label.shape
    acc = 0.0
    tot_predict = np.round(sigmoid(tot_logistic)).astype(int)

    for idx in range(n_drugs):
        print(f'  DrugID-{idx}-label, {label[:, idx].sum()}')
        print(f'  DrugID-{idx}-preds, {tot_predict[:, idx].sum()}')
        
    for i in range(n_cells):
        ac = 0.0
        for j in range(n_drugs):
            if label[i,j] == tot_predict[i,j]:
                ac += 1
        acc += ac / n_drugs
    acc /= n_cells

    auc_, auc_logit = 0.0, 0.0
    for i in range(n_drugs):
        auc_ += roc_auc_score(label[:,i], tot_predict[:,i])
        auc_logit += roc_auc_score(label[:,i], tot_logistic[:,i])

    f1s = []
    pre, rec, f1 = 0.0, 0.0, 0.0
    aupr, aupr_logistic = 0.0, 0.0
    for i in range(n_drugs):
        score = precision_recall_fscore_support(label[:,i], tot_predict[:,i], average='binary')
        pre += score[0]
        rec += score[1]
        f1 += score[2]
        f1s.append(score[2])
        precision, recall, _ = precision_recall_curve(label[:,i], tot_predict[:,i], pos_label=1)
        aupr += auc(recall, precision)
        precision, recall, _ = precision_recall_curve(label[:,i], tot_logistic[:,i], pos_label=1)
        aupr_logistic += auc(recall, precision)
    
    print("Total Accuracy %f\n" % (acc))
    print("Total AUC %f\n" % (auc_/n_drugs))
    print("Total AUC (logistic) %f\n" % (auc_logit/n_drugs))
    print("Total Precision %f\n" % (pre/n_drugs))
    print("Total Recall %f\n" % (rec/n_drugs))
    print("Total F1-score %f\n" % (f1/n_drugs))
    print("Total AUPR %f\n" % (aupr/n_drugs))
    print("Total AUPR (logistic) %f\n" % (aupr_logistic/n_drugs))
    for path in out_paths:
        with open(path, 'a') as f_rst:
            f_rst.write(f">>> [{args.dataset}] {args.embed}\n")
            f_rst.write("[#ofHidden] {} [LR] {} [Patience] {}\n".format(args.hidden, args.lr, args.patience))
            f_rst.write("Total Accuracy %f\n" % (acc))
            f_rst.write("Total AUC %f\n" % (auc_/n_drugs))
            f_rst.write("Total AUC (logistic) %f\n" % (auc_logit/n_drugs))
            f_rst.write("Total Precision %f\n" % (pre/n_drugs))
            f_rst.write("Total Recall %f\n" % (rec/n_drugs))
            f_rst.write("Total F1-score %f\n" % (f1/n_drugs))
            f_rst.write("Total AUPR %f\n" % (aupr/n_drugs))
            f_rst.write("Total AUPR (logistic) %f\n\n" % (aupr_logistic/n_drugs))


def get_mat_dataset():
    path = loadmat("../../Dataset/GDSC/cell_drug.mat")
    label = np.array(path['data'], dtype=np.float32)
    path = loadmat("../../Dataset/GDSC/index.mat")
    test_cell_indices = np.array(path['idx'], dtype=np.int32)
    test_cell_indices = np.squeeze(test_cell_indices)
    return label, test_cell_indices


def get_np_dataset(dataset, num_folds):
    label = np.load(f"../../Dataset/{dataset}/cell_drug.npy")
    with open(f"../../Dataset/{dataset}/test_cell_indices_k{num_folds}.json", "r") as f:
        test_cell_indices = json.load(f)
    return label.astype(np.float32), test_cell_indices


def get_indices_of_train_only_cells():
    with open(f"../../Dataset/CCLE/cell_list_train_only.txt", "r") as f:
        num_train_only = len(f.readlines())
    return num_train_only


class SimpleRAMP:
    now = None

    @classmethod
    def set_cls_node_list(cls, attr, dataset):
        setattr(cls, attr + 's', [])
        _list = getattr(cls, attr + 's')

        with open(f"../../Dataset/{dataset}/{attr}_list.txt", 'r') as f:
            for node in f.readlines():
                node = node.replace('\n', '')
                _list.append(node)

    @classmethod
    def set_cls_env(cls, args, base):
        cls.embed_name = args.embed.split('/')[-1]
        cls.epochs = args.epoch
        cls.patience = args.patience
        cls.early_stop_threshold = args.early_stop
        cls.lr = args.lr
        cls.batch = args.batch

        if args.extra:
            cls.set_cls_node_list('cell', f'GDSC_MERGED_{args.dataset}')
        else:
            cls.set_cls_node_list('cell', args.dataset)
        cls.set_cls_node_list('drug', args.dataset)

        out_path =  os.path.join(base, cls.embed_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cls.output_path = out_path

        if args.extra:
            _, folded_test_indices = get_np_dataset(args.dataset, args.num_folds)
            for fold, indices in folded_test_indices.items():
                for idx in range(len(indices)):
                    folded_test_indices[fold][idx] += 981
            cls.test_cell_indices = folded_test_indices
            cls.label, _ = get_np_dataset(f'GDSC_MERGED_{args.dataset}', 1)
        elif args.val_gdsc:
            # Set CCLE for test, and GDSC for train/val
            _, cls.test_cell_indices = get_mat_dataset()
            cls.label, extra_test_idx = get_np_dataset(args.dataset, args.num_folds)
            cls.extra_test_idx = extra_test_idx['fold-1']
        elif args.dataset == "GDSC":
            cls.label, cls.test_cell_indices = get_mat_dataset()
        else:
            cls.label, cls.test_cell_indices = get_np_dataset(args.dataset, args.num_folds)

    def __init__(self, embedding_path, hidden, k):
        self.load_folded_embedding(os.path.join(embedding_path, "cell_embedding-{}.txt".format(k)))
        self.set_indices(args, k)
        self.set_fold_env(k)
        self.fury = 0
        self.best = 0
        self.n_hidden = hidden

    def set_fold_env(self, k):
        if SimpleRAMP.now is False:
            SimpleRAMP.now = datetime.now().strftime("%m-%d-%H%M")

        self.fold_save_path = os.path.join(SimpleRAMP.output_path, str(k))
        if not os.path.exists(self.fold_save_path):
            os.makedirs(self.fold_save_path)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename="{}/train.log".format(self.fold_save_path), level=logging.DEBUG)
        logger = logging.getLogger()
        logger.info("Path for Models = {}".format(self.fold_save_path))
        self.logger = logger

    def load_folded_embedding(self, path):
        with open(path, 'r') as f:
            self.feats = np.zeros(tuple(map(int, f.readline().split())))
            embedding = f.readlines()
            for node in embedding:
                node = node.split()
                name = node[0]
                feat = np.array(node[1:], dtype=np.float32)
                self.feats[SimpleRAMP.cells.index(name)] = feat

    def set_indices(self, args, k):
        if args.val_gdsc:
            test_index = np.array(range((k - 1) * 98, k * 98))
            fold_test_idx = SimpleRAMP.test_cell_indices[test_index].tolist()
        elif args.dataset == 'GDSC':
            if k == args.num_folds:
                test_index = np.array(range((k - 1) * 98, 981))
            else:
                test_index = np.array(range((k - 1) * 98, k * 98))
            fold_test_idx = SimpleRAMP.test_cell_indices[test_index].tolist()
        else:
            fold_test_idx = SimpleRAMP.test_cell_indices[f'fold-{k}']

        if args.val_gdsc:
            ccle_test_start = SimpleRAMP.extra_test_idx[0]
            gdsc_indices = range(ccle_test_start)
            initial_fold_train_idx = np.setdiff1d(gdsc_indices, fold_test_idx)
        else:
            initial_fold_train_idx = np.setdiff1d(np.array(range(len(SimpleRAMP.cells))), fold_test_idx)

        np.random.shuffle(initial_fold_train_idx)
        split_point = int( 9 * len(initial_fold_train_idx) / 10  )
        self.fold_train_idx = initial_fold_train_idx[:split_point].tolist()
        self.fold_valid_idx = initial_fold_train_idx[split_point:].tolist()
        self.fold_test_idx = fold_test_idx
        print('# of training dataset', len(self.fold_train_idx))
        print('# of validation dataset', len(self.fold_valid_idx))
        print('# of test dataset', len(self.fold_test_idx))

    def create_model(self):
        X = Input((self.feats.shape[1], ))
        D_1 = Dense(self.n_hidden, kernel_regularizer=L2(1e-4), activation=tf.nn.leaky_relu)(X)
        D_2 = Dense(self.n_hidden, kernel_regularizer=L2(1e-4), activation=tf.nn.leaky_relu)(D_1)
        D_3 = Dense(self.n_hidden, kernel_regularizer=L2(1e-4), activation=tf.nn.leaky_relu)(D_2)
        Y = Dense(len(SimpleRAMP.drugs))(D_3)
        return Model(inputs=X, outputs=Y)

    @staticmethod
    def micro_f1(labels, logits):
        predicted = tf.math.round(tf.nn.sigmoid(logits))
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)

        true_pos = tf.math.count_nonzero(predicted * labels)
        false_pos = tf.math.count_nonzero(predicted * (labels - 1))
        false_neg = tf.math.count_nonzero((predicted - 1) * labels)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fmeasure = (2 * precision * recall) / (precision + recall)
        return tf.cast(fmeasure, tf.float32)

    def compute_loss(self, labels, logits):
        per_node_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        return tf.reduce_mean(tf.reduce_sum(per_node_losses, axis=-1))  # Compute mean loss _per node_

    def check_early_stopping(self, current_score):
        # Early stopping
        diff =  current_score - self.best
        if diff >= SimpleRAMP.early_stop_threshold:
            self.best = current_score
            self.fury = 0

            self.model.save("{}".format(self.fold_save_path))
            log = "Save the best model, so far."
            self.logger.debug(log)
        else:
            if self.fury == SimpleRAMP.patience:
                log = f"Stop training: Ran out of patience({SimpleRAMP.patience})"
                print(log)
                self.logger.debug(log)
                return True
            else:
                self.fury += 1
        return False

    def get_batch(self):
        idx = 0
        if self.current_batch + SimpleRAMP.batch > len(self.batch_map):
            num_to_append = SimpleRAMP.batch - (len(self.batch_map) - self.current_batch)
            idx = self.batch_map[self.current_batch : ]
            random.shuffle(self.batch_map)
            idx.extend(self.batch_map[ : num_to_append])
            self.current_batch = num_to_append
        else:
            idx = self.batch_map[ self.current_batch : self.current_batch + SimpleRAMP.batch ]
            self.current_batch += SimpleRAMP.batch
        return self.feats[idx], SimpleRAMP.label[idx]

    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.compute_loss(y_train, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        score = SimpleRAMP.micro_f1(y_train, predictions)
        return loss, score * 100

    def train(self, num_folds, k):
        # Params
        self.model = self.create_model()
        self.optimizer = Adam(learning_rate=SimpleRAMP.lr)
        self.model.summary(print_fn=self.logger.info)

        steps_per_epoch = len(self.fold_train_idx) / SimpleRAMP.batch
        train_time = time.time()
        ema_loss = 0
        self.batch_map = self.fold_train_idx.copy()
        self.current_batch = 0
        random.shuffle(self.batch_map)
        e = int(len(self.fold_train_idx) / SimpleRAMP.batch) * SimpleRAMP.epochs + 1
        for step in range(1, e):
            step_time = time.time()
            x_train, y_train = self.get_batch()
            loss, train_score = self.train_step(x_train, y_train)
            if step == 1:
                ema_loss = loss
            ema_loss = ema_loss * 0.99 + loss * 0.01

            validation = self.model.predict(self.feats[self.fold_valid_idx], batch_size=len(self.fold_valid_idx))
            valid_score = SimpleRAMP.micro_f1(SimpleRAMP.label[self.fold_valid_idx], validation) * 100

            log = "fold: {}/{}  step: {}  epoch: {}/{}  loss: {:.7f}  ema-loss: {:.7f}  train-score: {:.2f} %  valid-score: {:.2f} %" \
                  .format(k, num_folds, step, int(step/steps_per_epoch), SimpleRAMP.epochs, loss, ema_loss, train_score, valid_score)
            log = log + "  best-score: {} step, {:.2f} %".format(step - self.fury, self.best)
            print(log)
            self.logger.info(log)
            
            if self.check_early_stopping(valid_score):
                break

        log = "Training time: {:.4f}".format(time.time()-train_time)
        print(log)
        self.logger.info(log)

    def test(self, indices):
        return self.model.predict(self.feats[indices], batch_size=len(indices))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='[SimpleRAMP] Simple test tool of trained embeddings')
    parser.add_argument('--dataset', default='GDSC', type=str, help="Dataset to use among GDSC/CCLE/GC (GC=GDSC+CCLE)")
    parser.add_argument('--val_gdsc', action='store_true', 
            help='Use GDSC in training & validation, and use CCLE in testing')
    parser.add_argument('--extra', action='store_true', help='Use GDSC for extra training & validation, and use test with folded dataset')
    parser.add_argument('--embed', type=str, required=True, help="Path for embedding to evaluate")
    parser.add_argument('--batch', type=int, default=64, help="Mini-batch size")
    parser.add_argument('--epoch', type=int, default=200, help="Maximum number of epochs to train")
    parser.add_argument('--patience', type=int, default=50, 
            help="Number of steps to wait until early stopping while meeting the early stopping condition")
    parser.add_argument('--early_stop', type=float, default=0.005, 
            help="Threshold for early stop (stop if val-score diff less then 0.005 for `patience` steps)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning-rate")
    parser.add_argument('--hidden', type=int, default=1024, help="Number of hidden layers' unit")
    args = parser.parse_args()

    assert args.dataset in ['GDSC', 'CCLE', 'TCGA', 'GDSC_MERGED_CCLE', 'GDSC_MERGED_TCGA'], print('Use proper dataset.')
    if args.val_gdsc:
        assert args.dataset in ['GDSC_MERGED_CCLE', 'GDSC_MERGED_TCGA'], print('Mode for MERGED dataset.')

    args.num_folds = 0
    for f_name in os.listdir(args.embed):
        if 'total_embedding' in f_name:
            args.num_folds += 1

    base = f'../../SimpleTestResult/{args.dataset}'
    if not os.path.exists(base):
        os.makedirs(base)

    SimpleRAMP.set_cls_env(args, base)

    # for test
    ramp, test_idx, extra_preds = None, None, None
    preds = np.zeros_like(SimpleRAMP.label)
    for k in range(1, args.num_folds + 1):
        ramp = SimpleRAMP(args.embed, args.hidden, k)
        ramp.train(args.num_folds, k)

        test_idx = ramp.fold_test_idx
        preds[test_idx] = ramp.test(test_idx)
        if args.val_gdsc:
            extra_preds = ramp.test(SimpleRAMP.extra_test_idx)
        del ramp 

    label = SimpleRAMP.label
    #if args.dataset == 'CCLE':
    #    # The nodes don't have C-P edges can't be used as test cell
    #    num_train_only = get_indices_of_train_only_cells()
    #    preds = preds[num_train_only:]
    #    label = deepcopy(SimpleRAMP.label)[num_train_only:]
    if args.extra:
        preds = preds[981:]
        label = deepcopy(SimpleRAMP.label)[981:]
    if 'MERGED' in args.dataset:
        # GDSC cells are only used at training
        preds = preds[test_idx]
        label = deepcopy(SimpleRAMP.label)[test_idx]

    print(f"# of test cells: {label.shape[0]}")
    print(f"Label shape: {label.shape}")
    print(f"Preds shape: {preds.shape}")

    global_rst_path = os.path.join("../../SimpleTestResult", "entire_results.txt")
    #global_rst_path = os.path.join(base, "global_results.txt")
    target_rst_path = os.path.join(SimpleRAMP.output_path, "test_result.txt")
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    test_metrics(label.astype(int), preds, args, [global_rst_path, target_rst_path])
    if args.val_gdsc:
        print('>>> CCLE Test')
        label = SimpleRAMP.label.astype(int)
        label = label[SimpleRAMP.extra_test_idx]
        test_metrics(label, extra_preds, args, [global_rst_path, target_rst_path])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


