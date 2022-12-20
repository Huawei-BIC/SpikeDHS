import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.snndarts_retrain.LEAStereo import LEAStereo
import fitlog
import torch.nn.functional as F

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--experiment_description', type=str, help='description of experiment')
parser.add_argument('--fea_num_layers', type=int, default=8)
parser.add_argument('--fea_filter_multiplier', type=int, default=48)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--use_DGS', default=False, type=bool)
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

fitlog_debug = True 
if fitlog_debug:
    fitlog.debug()
else:
    fitlog.commit(__file__,fit_msg=args.experiment_description)
    log_path = "logs"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fitlog.set_log_dir(log_path)
    fitlog.create_log_folder()
    fitlog.add_hyper(args)
    # opt.fitlog_path = os.path.join(log_path,fitlog.get_log_folder())


CIFAR_CLASSES = 10


def main():

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)
  leastereo = LEAStereo(init_channels=3, args=args)
  model = leastereo
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  optimizer_b = torch.optim.Adam(model.arch_parameters(), lr=0.01, betas=(0.9,0.999))

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0
  for epoch in range(args.epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, optimizer_b, epoch)
    logging.info('train_acc %f', train_acc)
    fitlog.add_metric(train_acc,epoch,'train_top1')
    fitlog.add_metric(train_obj,epoch,'train_loss')

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    fitlog.add_metric(valid_acc,epoch,'valid_top1')
    fitlog.add_metric(valid_obj,epoch,'valid_loss')
    
    if valid_acc >= best_acc:
      best_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'weights.pt'))
      
    scheduler.step()
fitlog.finish()

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:4]
    else:
        this_str = this_str.split('.')
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append('[%s]'%value)
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append('.'+value)
    return ''.join(new_index)
    

def clamp(this, v_low=1, v_high=6):
    if this <= v_low:
        return v_low
    elif this >= v_high:
        return v_high
    else:
        return this

tem_b_all = list()

def train(train_queue, model, criterion, optimizer, optimizer_b, epoch):
  param = {'mode':'optimal'}

  N_diffb = 2
  N_diffb_mode = [i+(10-N_diffb) for i in range(N_diffb*2+1)]
  N_diffb_delta = [(i-10)*0.2 for i in N_diffb_mode]
  print(N_diffb_mode)
  print(N_diffb_delta)
  keys = [name for name, value in model.named_parameters() if 'alpha_diffb' in name]
  values = torch.FloatTensor([[0]*(N_diffb*2+1)]*len(keys))
  alpha_update_dict = dict(zip(keys, values))
  print(alpha_update_dict)
  
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  if args.use_DGS and epoch > 0 and epoch % 5 == 0:
    for step, (input, target) in enumerate(train_queue):
      if step > 200:
        break
      input = Variable(input).cuda()
      target = Variable(target).cuda()


      # copy optimal weight to s,m,b and init alpha value
      print('---------------copy optimal weight to s,m,b and init alpha value')
      model_weights = model.state_dict()
      for name,value in model.named_parameters():
          if 'snn_optimal' in name:
              for mode in N_diffb_mode:
                  replace_name = 'snn_%s'%mode
              # for replace_name in ['snn_s','snn_m', 'snn_b']:   
                  new_name = name.replace('snn_optimal',replace_name)
                  model_weights[new_name] = model_weights[name]
          if 'alpha_diffb' in name:
              model_weights[name] = 1e-3*torch.ones(len(N_diffb_mode))
      model.load_state_dict(model_weights)
      
      # update s,m,b weights
      print('---------------update s,m,b weights')
      for mode in N_diffb_mode:
      # for mode in ['small', 'middle', 'bigger']:
        param['mode'] = mode
        optimizer.zero_grad()
        optimizer_b.zero_grad()
        for name,value in model.named_parameters():
          if 'snn_%s'%mode not in name:
              value.requires_grad_(False)
          else:
              value.requires_grad_(True)
        logits, logits_aux = model(input, param)
        loss = criterion(logits, target)
        if args.auxiliary:
          loss_aux = criterion(logits_aux, target)
          loss += args.auxiliary_weight*loss_aux
        loss.backward()
        optimizer.step()

        # update alpha value
        print('---------------update alpha value')
        param['mode'] = 'combine'
        optimizer.zero_grad()
        optimizer_b.zero_grad()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        logits, logits_aux = model(input, param)
        loss = criterion(logits, target)
        if args.auxiliary:
          loss_aux = criterion(logits_aux, target)
          loss += args.auxiliary_weight*loss_aux
        loss.backward()
        optimizer_b.step()

          # record alpha value
        for key in alpha_update_dict:
            new_key = convert_str2index(key)
            # weight = convert_str2index(key,is_wight=True)
            # print('weight grad:',eval('model.%s'%weight).grad)
            print('alpha value:',eval('model.%s'%new_key))
            print('alpha softmax value:',F.softmax(eval('model.%s'%new_key)))
            print('alpha grad:',eval('model.%s'%new_key).grad)
            alpha_update_dict[key] += F.softmax(eval('model.%s'%new_key).clone().detach()).cpu()

    # update temp b
    tem_b_this = []
    for key in alpha_update_dict:
        new_key = convert_str2index(key)
        new_key_b = convert_str2index(key, is_b=True)
        max_index = torch.argmax(alpha_update_dict[key])
        tem_b_this.append(eval('model.%s'%new_key_b))
        exec('model.%s = clamp(model.%s+ N_diffb_delta[max_index])'%(new_key_b,new_key_b))
        print('new tem b:',eval('model.%s'%new_key_b))
        alpha_update_dict[key] = torch.FloatTensor([0]*(N_diffb*2+1))
    tem_b_all.append(tem_b_this)
    print('saving temb!!!!tem_b.npy')
    np.save(os.path.join(args.save,'tem_b.npy'),tem_b_all)



  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    optimizer_b.zero_grad()
    logits, logits_aux = model(input, param)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()

    
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    
    optimizer.step()


    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  param = {'mode':'optimal'}
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits, _ = model(input, param)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

