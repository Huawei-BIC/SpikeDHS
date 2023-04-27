import os
import sys
import numpy as np
import torch
import argparse
import pdb
import torch.nn.functional as F

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """
    return space

class Decoder(object):
    def __init__(self, alphas, steps):
        self._alphas = alphas
        self._steps = steps

    def genotype_decode(self):
        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self._steps)
        return gene_cell


def obtain_decode_args():
    parser = argparse.ArgumentParser(description="LEStereo Decoding..")
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'],
                        help='dataset name (default: sceneflow)') 
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    return parser.parse_args()

class Loader(object):
    def __init__(self, args):
        self.args = args
        # Resuming checkpoint
        assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
        assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        self._alphas_fea = checkpoint['module.feature.alphas']
        self.decoder_fea = Decoder(alphas=self._alphas_fea, steps=self.args.step)

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        return fea_genotype

def get_new_network_cell():
    args = obtain_decode_args()
    load_model = Loader(args)
    fea_genotype = load_model.decode_cell()
    print('Feature Net cell structure:', fea_genotype)

    dir_name = os.path.dirname(args.resume)
    fea_genotype_filename = os.path.join(dir_name, 'feature_genotype')
    np.save(fea_genotype_filename, fea_genotype)

    # fea_cell_name = os.path.join(dir_name, 'feature_cell_structure')  

if __name__ == '__main__':
    get_new_network_cell()