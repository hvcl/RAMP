import itertools
import os
from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import numpy as np

class Trainer():
    def __init__(self, args):
        '''
        Official Pytorch Code for RAMP: Response-Aware Multi-task Learning with Contrastive Regularization for Cancer Drug Response Prediction
        https://github.com/hvcl/RAMP
        '''
        self.args = args
        self.device = torch.device('cuda:0')
        self.model_names = ["F"]
        self.loss_names = ["CELoss", "SSCR", "Total_Loss"]
        
        self.in_ch = args["in_ch"]
        self.nparameter = args["nparameter"]
        self.ndrugs = args["ndrugs"]
        self.alpha = args["alpha"]
        self.dp_rate = args["dp_rate"]
        self.nsampling = args["nsampling"]
        self.F = Drug_Response_Predictor(self.in_ch, self.nparameter, self.ndrugs, self.dp_rate)
        self.F.to(self.device)

        self.loss = torch.nn.BCELoss(size_average=True, reduction='mean')
        self.optimizer_G = torch.optim.RAdam(itertools.chain(self.F.parameters()), 
                                             lr=args["lr"], betas=(0.9, 0.999)
        )

        self.print_networks()
        
    def set_input(self, data):
        self.x = data['x'].to(self.device)
        self.y = data['y'].to(self.device)
        self.mask = data['mask'].to(self.device)
        
    def forward(self):
        self.output, self.features = self.F(self.x, MCD=True)
        
    def backward_G(self): 
        #self.CELoss = torch.mean(-self.mask*(self.y*torch.log(self.output+1e-8)+(1-self.y)*torch.log(1-self.output+1e-8)))
        self.CELoss = torch.mean(self.mask*(torch.nn.functional.relu(self.output)-self.output*self.y+torch.log(1+torch.exp(-torch.abs(self.output)))))
        ## Calculate Label Similarity Matrix
        label = torch.cat([self.y[:,:,0], self.y[:,:,1]], dim=1)
        label_similarity = torch.matmul(label, torch.transpose(label,1,0))
        label_similarity /= self.ndrugs
        
        ## Calculate Feature Similarity Matrix
        normalized_feature = nn.functional.normalize(self.features, p=2.0, dim=1)
        feature_similarity = torch.matmul(normalized_feature, torch.transpose(normalized_feature,1,0))
        self.SSCR = self.alpha*torch.mean(-label_similarity*torch.log(feature_similarity))
        self.Total_Loss = self.CELoss + self.SSCR
        self.Total_Loss.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    # main loop
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def test(self):
        with torch.no_grad():
            self.forward()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
                
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def prediction(self, test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32, device=self.device, requires_grad=False)
        
        with torch.no_grad():
            output = 0.0
            for _ in range(0,self.nsampling):
                logits, _ = self.F(test_data, MCD=True)
                output += torch.nn.functional.sigmoid(logits)
            output /= self.nsampling
        return output.cpu().numpy()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        loss_names = self.loss_names
        for name in loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)
        return errors_ret
    
    
    def save_model(self):
        torch.save(self.F.state_dict(), self.args["result_dir"]+"/model/F.pth")
    
    def load_model(self):
        self.F.load_state_dict(torch.load(self.args["result_dir"]+"/model/F.pth"))
    
class Drug_Response_Predictor(nn.Module):
    def __init__(self, in_ch, nparameter, ndrugs, dp_rate):
        super(Drug_Response_Predictor, self).__init__()
        
        self.dp_rate = dp_rate
        self.shared_linear = nn.Sequential(nn.Linear(in_ch, nparameter, bias=True),
                                           nn.LeakyReLU(negative_slope=0.2))
        self.sn1 = nn.Sequential(nn.Linear(nparameter, nparameter, bias=True), 
                                 nn.LeakyReLU(negative_slope=0.2))
        self.sn2 = nn.Sequential(nn.Linear(nparameter, nparameter, bias=True), 
                                 nn.LeakyReLU(negative_slope=0.2))
        self.sn3 = nn.Linear(nparameter, ndrugs, bias=True)
        
        self.rn1 = nn.Sequential(nn.Linear(nparameter, nparameter, bias=True), 
                                 nn.LeakyReLU(negative_slope=0.2))
        self.rn2 = nn.Sequential(nn.Linear(nparameter, nparameter, bias=True), 
                                 nn.LeakyReLU(negative_slope=0.2))
        self.rn3 = nn.Linear(nparameter, ndrugs, bias=True)
        
    def forward(self, x, MCD=True):
        shared_feature = self.shared_linear(x)
        
        y_sn = self.sn1(shared_feature)
        skip_sn_feature = y_sn
        y_sn = nn.functional.dropout(y_sn, training=MCD)
        y_sn = self.sn2(y_sn)
        y_sn = nn.functional.dropout(y_sn, training=MCD)
        y_sn = self.sn3(y_sn)
        
        y_rn = self.rn1(shared_feature)
        skip_rn_feature = y_rn
        y_sn = nn.functional.dropout(y_sn, training=MCD)
        y_rn = self.rn2(y_rn)
        y_sn = nn.functional.dropout(y_sn, training=MCD)
        y_rn = self.rn3(y_rn)
        
        return torch.cat([torch.unsqueeze(y_sn, 2), torch.unsqueeze(y_rn,2)], dim=2), torch.cat([shared_feature, skip_sn_feature, skip_rn_feature], dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)