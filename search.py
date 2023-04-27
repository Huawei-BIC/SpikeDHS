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
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
# from architect import Architect
from models.snndarts_search.build_model import AutoStereo
import fitlog


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
# parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--experiment_description', type=str, help='description of experiment')


######### LEStereo params ##################
parser.add_argument('--fea_num_layers', type=int, default=8)
parser.add_argument('--fea_filter_multiplier', type=int, default=8)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--drop_rate', default=0.5, type=float)
parser.add_argument('--fitlog_path',type=str,default='debug')   

######### search params ##################
parser.add_argument('--arch_lr', default=0.001, type=float)

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


fitlog_debug = False
# fitlog_debug = True
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

if args.CIFAR10 or args.DVSCIFAR10:
    CIFAR_CLASSES = 10
elif args.CIFAR100:
    CIFAR_CLASSES = 100
else:
    raise NotImplementedError



def main():
  
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  model = AutoStereo(args.init_channels, args = args)
  model = model.cuda()
  model = torch.nn.DataParallel(model)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.module.feature.weight_parameters(), 
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  architect_optimizer = torch.optim.Adam(model.module.feature.arch_parameters(),
                                              lr=args.arch_lr, betas=(0.9, 0.999),
                                              weight_decay=args.arch_weight_decay)


  if args.CIFAR10:
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  elif args.CIFAR100:
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  elif args.DVSCIFAR10:
    train_data, valid_data = utils.build_dvscifar(path=args.data)


  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train)) # 25000

  if args.CIFAR10 or args.CIFAR100:
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=14)   # 25000

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=14)   # 25000

  elif args.DVSCIFAR10:
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        pin_memory=True, num_workers=2)   # 9000 

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=20,
        pin_memory=True, num_workers=2)   # 1000

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  for epoch in range(args.epochs):
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, architect_optimizer, lr, epoch)
    logging.info('train_acc %f', train_acc)
    fitlog.add_metric(train_acc,epoch,'train_top1')
    fitlog.add_metric(train_obj,epoch,'train_loss')

    # validation
    if args.epochs-epoch <=1: 
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        fitlog.add_metric(valid_acc,epoch,'valid_top1')
        fitlog.add_metric(valid_obj,epoch,'valid_loss')

    utils.save(model, os.path.join(args.save, 'epoch_%s.pt'%epoch))
    scheduler.step()

fitlog.finish()


def train(train_queue, valid_queue, model, criterion, optimizer, architect_optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input = Variable(input, requires_grad=False).to(torch.cuda.current_device())
    target = Variable(target, requires_grad=False).to(torch.cuda.current_device())

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).to(torch.cuda.current_device())
    target_search = Variable(target_search, requires_grad=False).to(torch.cuda.current_device())

    optimizer.zero_grad()
    architect_optimizer.zero_grad()
    
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    architect_optimizer.zero_grad()

    ##### search forward #####
    logits = model(input_search)
    loss = criterion(logits, target_search)
    loss.backward()
    architect_optimizer.step()
    optimizer.zero_grad()
    architect_optimizer.zero_grad()
    ##### END search forward #####

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if fitlog_debug:
      break
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).to(torch.cuda.current_device())
      target = Variable(target).to(torch.cuda.current_device())

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if fitlog_debug:
        break
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

