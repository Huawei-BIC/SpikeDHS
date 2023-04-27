import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from time import time
import numpy as np
from models.snndarts_search.build_model_2d import AutoFeature
from models.snndarts_search.decoding_formulas import network_layer_to_space
class AutoStereo(nn.Module):
    def __init__(self, init_channels=3, args=None):
        super(AutoStereo, self).__init__()
        p=0.0
        network_path_fea = [0,1,2,3]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        cell_arch_fea = [[1, 1],
                            [0, 1],
                            [3, 2],
                            [2, 1],
                            [7, 1],
                            [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)

        self.feature  = AutoFeature(init_channels, args=args)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(192, 10)

    def update_p(self):
        self.feature.p = self.p
        self.feature.update_p()

        
    def forward(self, input, timestamp=6): 
        
        param = {'snn_output':'mem'}
        cost_all = []
        logits = None
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out = self.feature(input, param) 
            pooling_out = self.global_pooling(feature_out)
            logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1))

            if logits is None:
                logits = []
            logits.append(logits_buf)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp
        return logits
