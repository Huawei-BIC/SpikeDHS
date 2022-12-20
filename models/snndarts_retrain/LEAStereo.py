import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.snndarts_search.SNN import *
from models.snndarts_search.decoding_formulas import network_layer_to_space
from models.snndarts_retrain.new_model_2d import newFeature
import time

class LEAStereo(nn.Module):
    def __init__(self, init_channels=3, args=None):
        super(LEAStereo, self).__init__()
        p=0.0
        network_path_fea = [0,0,1,1,1,2,2,2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        cell_arch_fea = [[1, 1],
                            [0, 1],
                            [3, 2],
                            [2, 1],
                            [7, 1],
                            [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)

        self.feature = newFeature(init_channels,network_arch_fea, cell_arch_fea, args=args)
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(576, 10)

        
    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' in name]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'alpha_diffb' not in name]

    def forward(self, input, param): 
        param['snn_output'] = 'mem'

        
        logits = None
        logits_aux_list = []
        timestamp = 6
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux = self.feature(input, param) 
            logits_aux_list.append(logits_aux)

 
            pooling_out = self.global_pooling(feature_out) 
            logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1)) 

            if logits is None:
                logits = []
            logits.append(logits_buf)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final
        else:
            return logits, None


def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])

