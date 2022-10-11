import os
import argparse
from multiprocessing import Process
import multiprocessing as mp
import ctypes as c

import numpy as np
import scipy.io

from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import RandomOverSampler


def multi_task_prediction(train_data, train_label, test_data):
    preds = np.zeros([test_data.shape[0], train_label.shape[1]], dtype=np.float32)
    for drug_idx in range(train_label.shape[1]):
        #parameters = {'C':[1.0, 5.0, 10.0, 100.0],
        #              'kernel':['linear'],
        #              'gamma':['scale','auto'],
        #              }
        #              #'probability':[True]}
        #searcher = GridSearchCV(SVC(), parameters, scoring='f1_macro')
        ros = RandomOverSampler(random_state=1)
        x, y = ros.fit_resample(train_data, train_label[:, drug_idx])
        #searcher.fit(x,y)
        #print(searcher.best_params_)
        #
        #svc = SVC(**searcher.best_params_)
        svc = SVC(C=1.0, gamma='auto', probability=True)
        svc.fit(x, y)
        out = svc.predict_proba(test_data)
        preds[:, drug_idx] = out[:,1]
        #out = svc.predict(test_data)
        #preds[:, drug_idx] = out
    return preds


def get_gdsc_dataset():
    path = scipy.io.loadmat("../Dataset/GDSC/cell_drug.mat")
    label = np.array(path['data'], dtype=np.int32)
    path = scipy.io.loadmat("../Dataset/GDSC/index.mat")
    test_cell_indices = np.array(path['idx'], dtype=np.int32)
    test_cell_indices = np.squeeze(test_cell_indices)
    return label, test_cell_indices


def get_ccle_dataset(num_folds):
    import json

    label = np.load("../Dataset/CCLE/cell_drug.npy")
    with open(f"../Dataset/CCLE/test_cell_indices_k{num_folds}.json", "r") as f:
        test_cell_indices = json.load(f)
    return label, test_cell_indices


def evaluate_multi(cell_list, label, test_cell_indices, args, k, out_que):
    num_cells, num_drugs = label.shape
    print(f"Start fold-{k} evaluation..")
    with open(f"{args.embed}/cell_embedding-{k}.txt", 'r') as f_embed:
        embedding = f_embed.readlines()
        features = np.zeros([num_cells, len(embedding[1].split(" ")) - 1])
        for idx in range(1, len(embedding)):
            node = embedding[idx].split(" ")
            name = node[0]
            feat = np.array(node[1:], dtype=np.float32)
            features[cell_list.index(name), :] = feat

    if args.dataset == 'GDSC':
        if k == args.num_folds:
            test_index = np.array(range((k - 1) * 98, 981))
        else:
            test_index = np.array(range((k - 1) * 98, k * 98))
        test_index = test_cell_indices[test_index]
    else:
        test_index = test_cell_indices[f'fold-{k}']

    train_index = np.setdiff1d(np.array(range(num_cells)), test_index)
    train_data = features[train_index]
    train_label = np.array(label[train_index], dtype=np.float32)
    test_data = features[test_index]

    fold_pred = multi_task_prediction(train_data, train_label, test_data)
    out_que.put((k, fold_pred))


def evaluate(cell_list, label, test_cell_indices, args):
    num_cells, num_drugs = label.shape
    predicts = np.zeros(label.shape, dtype=np.float32)
    for k in range(1, args.num_folds + 1):
        print(f"Start fold-{k} evaluation..")
        features = np.zeros([num_cells, len(embedding[1].split(" ")) - 1])
        with open(f"{args.embed}/cell_embedding-{k}.txt", 'r') as f_embed:
            embedding = f_embed.readlines()
            for idx in range(1, len(embedding)):
                node = embedding[idx].split(" ")
                name = node[0]
                feat = np.array(node[1:], dtype=np.float32)
                features[cell_list.index(name), :] = feat

        if args.dataset == 'GDSC':
            if k == args.num_folds:
                test_index = np.array(range((k - 1) * 98, 981))
            else:
                test_index = np.array(range((k - 1) * 98, k * 98))
            test_index = test_cell_indices[test_index]
        else:
            test_index = test_cell_indices[f'fold-{k}']

        train_index = np.setdiff1d(np.array(range(num_cells)), test_index)
        train_data = features[train_index]
        train_label = np.array(label[train_index], dtype=np.float32)
        test_data = features[test_index]

        fold_pred = multi_task_prediction(train_data, train_label, test_data)
        predicts[test_index,:] = fold_pred
    return predicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[RAMP] Simple test tool of trained embeddings')
    parser.add_argument('--dataset', default='GDSC', type=str, help="Dataset to use among GDSC/CCLE/GC (GC=GDSC+CCLE)")
    parser.add_argument('--embed', type=str, required=True, help="Path for embedding to evaluate")
    parser.add_argument('--multi', action='store_true', help='Use Multi-Processing')
    args = parser.parse_args()

    args.num_folds = 0
    for f_name in os.listdir(args.embed):
        if 'total_embedding' in f_name:
            args.num_folds += 1

    with open(f"../Dataset/{args.dataset}/cell_list.txt", 'r') as f:
        cell_list = []
        for cell in f.readlines():
            cell = cell.replace('\n', '')
            cell_list.append(cell)
    
    with open(f"../Dataset/{args.dataset}/drug_list.txt", 'r') as f:
        drug_list = []
        for drug in f.readlines():
            d = drug.replace('\n', '')
            drug_list.append(d)
    
    if args.dataset == "GDSC":
        label, test_cell_indices = get_gdsc_dataset()
    elif args.dataset == "CCLE":
        label, test_cell_indices = get_ccle_dataset(args.num_folds)
    
    if args.multi:
        out_que = mp.Queue()
        workers = []
        for k in range(1, args.num_folds + 1):
            proc = Process(target=evaluate_multi, args=(cell_list, label, test_cell_indices, args, k, out_que))
            workers.append(proc)
            proc.start()
        for worker in workers:
            worker.join()
        print('All precessings done.')

        out = [out_que.get() for _ in range(args.num_folds)]
        preds = np.zeros(label.shape, dtype=np.float32)
        for k, pred in out:
            if args.dataset == 'GDSC':
                if k == args.num_folds:
                    test_index = np.array(range((k - 1) * 98, 981))
                else:
                    test_index = np.array(range((k - 1) * 98, k * 98))
                test_index = test_cell_indices[test_index]
            else:
                test_index = test_cell_indices[f'fold-{k}']
            preds[test_index] = pred
        preds = np.round(preds)
    else:
        preds = np.round(evaluate(cell_list, label, test_cell_indices, args))

    with open('./test_result.txt', 'a') as f:
        f.write('>> ' + args.embed + '\n')
        score = precision_recall_fscore_support(label, preds, average='macro')
        f.write(f'prec: {score[0]:.4f}, recall: {score[1]:.4f}, f1: {score[2]:.4f}\n')

