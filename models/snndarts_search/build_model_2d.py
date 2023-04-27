import torch.nn as nn
import torch.nn.functional as F
import models.snndarts_search.cell_level_search_2d as cell_level_search
from models.snndarts_search.genotypes_2d import PRIMITIVES
from models.snndarts_search.operations_2d import *
from models.snndarts_search.decoding_formulas import Decoder
import pdb

class DispEntropy(nn.Module):
    def __init__(self, maxdisp):
        super(DispEntropy, self).__init__()
        self.softmax = nn.Softmin(dim=1)
        self.maxdisp = maxdisp

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        e = torch.sum(-F.softmax(x,dim=1) * F.log_softmax(x,dim=1),1)
        m = 1.0- torch.isnan(e).type(torch.cuda.FloatTensor)
        x = e*m
        x = self.softmax(x)
        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, 260, 346], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x


class AutoFeature(nn.Module):
    def __init__(self, frame_rate, args ,p=0.0):
        super(AutoFeature, self).__init__()
        cell=cell_level_search.Cell
        self.cells = nn.ModuleList()
        self.p = p
        self._num_layers = args.fea_num_layers
        self._step = args.fea_step
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier
        self._initialize_alphas_betas()
        self.args = args
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        self.stem0 = ConvBR(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1)

        '''
            cell(step, block, prev_prev, prev_down, prev_same, prev_up, filter_multiplier)

            prev_prev, prev_down etc depend on tiers. If cell is in the first tier, then it won`t have prev_down.
            If cell is in the second tier, prev_down should be filter_multiplier *2, if third, then *4.(filter_multiplier is an absolute number.)
        '''

        for i in range(self._num_layers):
            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier,self.p)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2,self.p)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, None,
                             self._filter_multiplier,self.p)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2,self.p)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4,self.p)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, None,
                             self._filter_multiplier,self.p)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2,self.p)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4,self.p)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8,self.p)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, None,
                             self._filter_multiplier,self.p)
                
                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2,self.p)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4,self.p)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8,self.p)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, None,
                             self._filter_multiplier,self.p)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2,self.p)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4,self.p)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8,self.p)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.last_3  = ConvBR(self._num_end , self._num_end, 1, 1, 0, bn=False, relu=False) 
        self.last_6  = ConvBR(self._num_end*2 , self._num_end,    1, 1, 0)  
        self.last_12 = ConvBR(self._num_end*4 , self._num_end*2,  1, 1, 0)  
        self.last_24 = ConvBR(self._num_end*8 , self._num_end*4,  1, 1, 0) 

        self.down_sample1 =  ConvBR(self._num_end , self._num_end*2,  3, 2, 1)
        self.down_sample2 =  ConvBR(self._num_end*2 , self._num_end*4,  3, 2, 1)
        self.down_sample3 =  ConvBR(self._num_end*4 , self._num_end*8,  3, 2, 1)
        self.down_sample4 =  ConvBR(self._num_end*8 , self._num_end*8,  1, 1, 0, bn=False, relu=False) 

    def forward(self, x, param):
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []
        stem0 = self.stem0(x)
        self.level_3.append(stem0)
        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 2)
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:
            #print('more than 1 gpu used!')
            img_device = torch.device("cuda:{}".format(self.args.device))
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)
            
            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0].to(device=img_device), dim=-1)

                elif layer == 1:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0].to(device=img_device), dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0].to(device=img_device), dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0].to(device=img_device), dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:1] = F.softmax(self.betas[layer][3][:1].to(device=img_device), dim=-1) * (1/2)

        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0], dim=-1)

                elif layer == 1:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0] = F.softmax(self.betas[layer][0], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:1] = F.softmax(self.betas[layer][3][:1], dim=-1) * (1/2)


        for layer in range(self._num_layers):

            if layer == 0:
                level3_new, = self.cells[count](None, None, self.level_3[-1], None, normalized_alphas, param)
                count += 1
                level6_new, = self.cells[count](None, self.level_3[-1], None, None, normalized_alphas, param)
                count += 1

                level3_new = normalized_betas[layer][0][0] * level3_new
                level6_new = normalized_betas[layer][0][1] * level6_new
                self.level_3.append(level3_new)
                self.level_6.append(level6_new)

            elif layer == 1:
                level3_new_1, = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               None,
                                                               normalized_alphas, param)
                count += 1
                level3_new = normalized_betas[layer][0][0] * level3_new_1

                level6_new_1, level6_new_2 = self.cells[count](None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               None,
                                                               normalized_alphas, param)
                count += 1
                level6_new = normalized_betas[layer][0][1] * level6_new_1 + normalized_betas[layer][1][0] * level6_new_2

                level12_new, = self.cells[count](None,
                                                 self.level_6[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas, param)
                # print("layer1,level12",level12_new.size()) 1 48 22 29
                level12_new = normalized_betas[layer][1][1] * level12_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)

            elif layer == 2:
                level3_new_1, = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               None,
                                                               normalized_alphas, param)
                count += 1
                level3_new = normalized_betas[layer][0][0] * level3_new_1

                level6_new_1, level6_new_2,  = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             None,
                                                                             normalized_alphas, param)
                count += 1
                level6_new = normalized_betas[layer][0][1] * level6_new_1 + normalized_betas[layer][1][0] * level6_new_2
                level12_new_1, level12_new_2 = self.cells[count](None,
                                                                 self.level_6[-1],
                                                                 self.level_12[-1],
                                                                 None,
                                                                 normalized_alphas, param)
                # print("layer2,level12_1",level12_new_1.size()) 1 48 22 29
                # print("layer2,level12_2",level12_new_2.size()) 1 48 22 29
                count += 1
                level12_new = normalized_betas[layer][1][1] * level12_new_1 + normalized_betas[layer][2][0] * level12_new_2

                level24_new, = self.cells[count](None,
                                                 self.level_12[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas, param)
                # print("layer2,level24_1",level24_new.size()) 1 96 11 15
                level24_new = normalized_betas[layer][2][1] * level24_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            elif layer == 3:
                level3_new_1, = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               None,
                                                               normalized_alphas, param)
                count += 1
                level3_new = normalized_betas[layer][0][0] * level3_new_1

                level6_new_1, level6_new_2,  = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             None,
                                                                             normalized_alphas, param)
                count += 1
                level6_new = normalized_betas[layer][0][1] * level6_new_1 + normalized_betas[layer][1][0] * level6_new_2

                level12_new_1, level12_new_2,  = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                None,
                                                                                normalized_alphas, param)
                count += 1
                level12_new = normalized_betas[layer][1][1] * level12_new_1 + normalized_betas[layer][2][0] * level12_new_2 
                level24_new_1, level24_new_2 = self.cells[count](None,
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas, param)
                count += 1
                level24_new = normalized_betas[layer][2][1] * level24_new_1 + normalized_betas[layer][3][0] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            else:
                level3_new_1, = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               None,
                                                               normalized_alphas, param)
                count += 1
                level3_new = normalized_betas[layer][0][0] * level3_new_1

                level6_new_1, level6_new_2, = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             None,
                                                                             normalized_alphas, param)
                count += 1

                level6_new = normalized_betas[layer][0][1] * level6_new_1 + normalized_betas[layer][1][0] * level6_new_2
                level12_new_1, level12_new_2, = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                None,
                                                                                normalized_alphas, param)
                count += 1
                level12_new = normalized_betas[layer][1][1] * level12_new_1 + normalized_betas[layer][2][0] * level12_new_2 
                level24_new_1, level24_new_2 = self.cells[count](self.level_24[-2],
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas, param)
                count += 1
                level24_new = normalized_betas[layer][2][1] * level24_new_1 + normalized_betas[layer][3][0] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            self.level_3 = self.level_3[-2:]
            self.level_6 = self.level_6[-2:]
            self.level_12 = self.level_12[-2:]
            self.level_24 = self.level_24[-2:]
        
        # downsample
        result_3  = self.down_sample4(self.down_sample3(self.down_sample2(self.down_sample1(self.level_3[-1]))))
        result_6  = self.down_sample4(self.down_sample3(self.down_sample2(self.level_6[-1])))
        result_12 = self.down_sample4(self.down_sample3(self.level_12[-1]))
        if len(self.level_24) != 0:
            result_24 = self.down_sample4(self.level_24[-1])     
            sum_feature_map =result_3 + result_6 + result_12 + result_24
        
        else:
            sum_feature_map = result_3+result_6+result_12
        return sum_feature_map


    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 2)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

