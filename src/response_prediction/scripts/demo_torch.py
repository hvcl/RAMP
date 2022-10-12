PATH = "RAMP" # Absolute Path of RAMP
import numpy as np
import sys
import scipy.io
import time
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
sys.path.append(PATH+"/src/response_prediction")
from utils.utils import get_performance, TqdmLoggingHandler
from torch.utils.data import DataLoader
from models.RAMP_torch import Trainer

class Dataset_Pipeline(torch.utils.data.Dataset):
    def __init__(self, data, label, mask, batch_size):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
        self.mask = torch.from_numpy(mask)
        self.batch_size = batch_size
        self.data_len = len(self.data)
    def __getitem__(self, index):
        idx = index % self.data_len
        x = self.data[idx]
        y = self.label[idx]
        mask = self.mask[idx]
        return {"x":x, "y":y, "mask":mask}
        
    def __len__(self):
        return max(self.data_len, self.batch_size)

def multi_task_prediction(train_data, train_label, test_data, alpha, dp_rate, ignore_idx):
    args = dict()
    args["in_ch"] = 64 # 
    args["nparameter"] = 2048
    args["ndrugs"] = 265
    args["alpha"] = alpha
    args["dp_rate"] = dp_rate
    args["nsampling"] = 500
    args["lr"] = 1e-4 
    args["epoch"] = 400
    args["batch_size"] = 64
    args["iter"] = (len(train_data) // args["batch_size"])+1
    log = logging.getLogger(__name__)
    log.handlers = []
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())
    
    mask = np.array(np.concatenate((ignore_idx[:,:,np.newaxis], ignore_idx[:,:,np.newaxis]), axis=2), dtype=np.float32)
    mask = 1-mask

    train_dataset = Dataset_Pipeline(train_data, train_label, mask, args["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=2)
    model = Trainer(args)
    
    # Logger
    
    # Start Training
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    train_generator = iter(cycle(train_loader))

    pbar = tqdm(range(0, args["epoch"]))
    
    for epoch in pbar:
        model.train()
        total_loss = {"CELoss":0.0, "SSCR":0.0, "Total_Loss":0.0}
        for iters in range(0, args["iter"]):
            data = next(train_generator)
            model.set_input(data)
            model.optimize_parameters()
        losses = model.get_current_losses()
        for i, key in enumerate(losses):
            total_loss[key] += losses[key].detach().cpu().numpy()
        pbar.set_postfix(total_loss)
            
    model.eval()
    
    predict = model.prediction(test_data)

    return predict
    
def cross_validation(data, label, inner_cv, alpha, dp_rate, ignore_idx):
    np.random.seed(2022)
    n_samples, n_drugs, _ = np.shape(label)
    index = np.random.permutation(n_samples)
    
    predict = np.zeros([n_samples, n_drugs, 2], dtype=np.float32)
    for k in range(0,inner_cv):
        if k+1 == inner_cv:
            start_idx = k * (n_samples // inner_cv) 
            end_idx = n_samples
        else:
            start_idx = k * (n_samples // inner_cv)
            end_idx = (k+1) * (n_samples // inner_cv)
        valid_idx = index[start_idx:end_idx]
        train_idx = np.setdiff1d(np.array(range(0,n_samples)), valid_idx)
        train_data = data[train_idx]
        train_label = label[train_idx]
        validation_data = data[valid_idx]
        validation_label = label[valid_idx]
        predict[valid_idx,:,:] = multi_task_prediction(train_data, train_label, validation_data, alpha, dp_rate, ignore_idx[train_idx])
    prob = np.array(predict[:,:,1], dtype=np.float32)
    result = np.zeros_like(prob, dtype=np.int32)
    for i in range(0, n_drugs):
        for j in range(0, n_samples):
            if predict[j,i,1] >= predict[j,i,0]:
                result[j,i] = 1
            else:
                result[j,i] = 0
    output = get_performance(label[:,:,1], prob, result, ignore_idx)
    return output

if __name__ == '__main__':

        # Load cell list
    cell_file = open(PATH + "/data/GDSC/cell_list.txt", 'r')
    cell_list = []
    
    for line in cell_file:
        cell_name = line.replace('\n', '')
        cell_list.append(cell_name)
    n_cells = len(cell_list)
    
    # Load drug list
    drug_file = open(PATH+"/data/GDSC/drug_list.txt", 'r')
    drug_list = []
    for line in drug_file:
        drug_name = line.replace('\n', '')
        drug_list.append(drug_name)
    n_drugs = len(drug_list)
    
    # Setting indices for 10-fold outer cross validation
    index_path = scipy.io.loadmat(PATH + "/data/GDSC/index.mat")
    kf = np.array(index_path['idx'], dtype=np.int32)
    kf = np.squeeze(kf)

    # A (sensitive response drug <-> cell line) network to label array
    label = np.zeros([n_cells,n_drugs,2], dtype=np.float32)
    label_path = open(PATH + "/data/GDSC/GDSC.Sensitive.Cell-Drug.txt", 'r')
    for line in label_path:
        line = line.replace('\n', '')
        words = line.split("\t")
        cell_name = words[0]
        drug_name = words[1]
        try:
            label[cell_list.index(cell_name), drug_list.index(drug_name),1] = 1
        except:
            print("{} cell lines isn't included in cell line lists or {} drug isn't included in drug lists".format(cell_name, drug_name))
            continue
    # A (resistant response drug <-> cell line) network to label array
    label_path = open(PATH + "/data/GDSC/GDSC.Resistant.Cell-Drug.txt", 'r')
    for line in label_path:
        line = line.replace('\n', '')
        words = line.split("\t")
        cell_name = words[0]
        drug_name = words[1]
        try:
            label[cell_list.index(cell_name), drug_list.index(drug_name),0] = 1
        except:
            print("{} cell lines isn't included in cell line lists or {} drug isn't included in drug lists".format(cell_name, drug_name))
            continue

    # Ignore indices represent unknown links between cell lines and drugs
    ignore_idx = np.zeros([n_cells,n_drugs], dtype=bool)
    for i in range(0,n_cells):
        for j in range(0,n_drugs):
            if label[i,j,0] == 0 and label[i,j,1] == 0:
                ignore_idx[i,j] = True

    final_predict = np.zeros([n_cells, n_drugs,2], dtype=np.float32)
    
    # Experimental Setting
    best_alpha_list = []
    best_dp_rate_list = []
    n_channels = 64
    Outer_fold = 10
    Inner_fold = 10
    search_alpha_list = [10.0, 15.0, 20.0, 25.0, 30.0] # Average alpha instead of search list
    search_dp_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5] # Average dp rate instead of search list
    #search_epoch_list = [400, 500, 600] 
    embedding_type = "rans"
    
    for k in range(0,Outer_fold): 
        Embedding = open(PATH + "/results/GDSC/embeddings/{}/cell_embedding-{}.txt".format(embedding_type, k+1), 'r')
        
        data = np.zeros([n_cells,n_channels], dtype=np.float32)
        
        for flag, line in enumerate(Embedding):
            if flag == 0:
                continue
            words = line.split(" ")
            tag = str(words[0])
            embedding_vector = np.array(words[1:], dtype=np.float32)
            try:
                data[cell_list.index(tag),:] = embedding_vector
            except:
                continue
        
        ### Outer Cross Validation Setting
        if k == Outer_fold-1:
            start_idx = k*(n_cells // Outer_fold)
            end_idx = n_cells
        else:
            start_idx = k*(n_cells // Outer_fold)
            end_idx = (k+1)*(n_cells // Outer_fold)
         
        test_index = kf[start_idx:end_idx]
        train_index = np.setdiff1d(np.array(range(0,n_cells)), test_index)
        train_data = data[train_index]
        train_label = np.array(label[train_index], dtype=np.float32)
        test_data = data[test_index]
        test_label = label[test_index]
        
        ### Hyper-parameters Optimization
        best_performance = 0.0
        best_alpha = 1.0
        best_dp_rate = 0.0

        for alpha in search_alpha_list:
            for dp_rate in search_dp_rate_list:
                performance = cross_validation(train_data, train_label, Inner_fold, alpha, dp_rate, ignore_idx[train_index])
                accuracy = sum(performance[0])*100
                auc = sum(performance[1])*100
                aupr = sum(performance[2])*100
                fscore = sum(performance[5])*100
                tot_performance = accuracy+auc+aupr+fscore
                if tot_performance > best_performance:
                    best_performance = tot_performance
                    best_alpha = alpha
                    best_dp_rate = dp_rate
                    print("Accuracy: %f, AUC: %f, AUPR: %f, F1-score: %f" % (accuracy/n_drugs, auc/n_drugs, aupr/n_drugs, fscore/n_drugs))
        
        predict = multi_task_prediction(train_data, train_label, test_data, best_alpha, best_dp_rate, ignore_idx[train_index])
        best_alpha_list.append(best_alpha)
        best_dp_rate_list.append(best_dp_rate)
        final_predict[test_index] = predict

    
    scipy.io.savemat(PATH + "/results/predictions/RAMP.mat", {"GDSC_data":data, "label":label, "pred":final_predict, "alpha":best_alpha_list, "dp":best_dp_rate_list})
        
    
