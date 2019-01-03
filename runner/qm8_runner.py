from __future__ import (division, print_function)
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper

logger = get_logger('exp_logger')

__all__ = ['QM8Runner']


class QM8Runner(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.meta_data = pickle.load(open(self.dataset_conf.meta_data_path, 'rb'))
    self.const_factor = self.meta_data['std'].reshape(1, -1)

  def train(self):
    # create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(
        self.config, split='train')
    dev_dataset = eval(self.dataset_conf.loader_name)(self.config, split='dev')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=False,
        num_workers=self.train_conf.num_workers,
        collate_fn=dev_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(
          params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_steps,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.train_conf.resume_model, optimizer=optimizer)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # Training Loop
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # validation
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []

        for data in tqdm(dev_loader):
          if self.use_gpu:
            data['node_feat'], data['node_mask'], data['label'] = data_to_gpu(
                data['node_feat'], data['node_mask'], data['label'])

            if self.model_conf.name == 'LanczosNet':
              data['L'], data['D'], data['V'] = data_to_gpu(
                  data['L'], data['D'], data['V'])
            elif self.model_conf.name == 'GraphSAGE':
              data['nn_idx'], data['nonempty_mask'] = data_to_gpu(
                  data['nn_idx'], data['nonempty_mask'])
            elif self.model_conf.name == 'GPNN':
              data['L'], data['L_cluster'], data['L_cut'] = data_to_gpu(
                  data['L'], data['L_cluster'], data['L_cut'])
            else:
              data['L'] = data_to_gpu(data['L'])[0]

          with torch.no_grad():
            if self.model_conf.name == 'AdaLanczosNet':
              pred, _ = model(
                  data['node_feat'],
                  data['L'],
                  label=data['label'],
                  mask=data['node_mask'])
            elif self.model_conf.name == 'LanczosNet':
              pred, _ = model(
                  data['node_feat'],
                  data['L'],
                  data['D'],
                  data['V'],
                  label=data['label'],
                  mask=data['node_mask'])
            elif self.model_conf.name == 'GraphSAGE':
              pred, _ = model(
                  data['node_feat'],
                  data['nn_idx'],
                  data['nonempty_mask'],
                  label=data['label'],
                  mask=data['node_mask'])
            elif self.model_conf.name == 'GPNN':
              pred, _ = model(
                  data['node_feat'],
                  data['L'],
                  data['L_cluster'],
                  data['L_cut'],
                  label=data['label'],
                  mask=data['node_mask'])
            else:
              pred, _ = model(
                  data['node_feat'],
                  data['L'],
                  label=data['label'],
                  mask=data['node_mask'])

          curr_loss = (
              pred - data['label']).abs().cpu().numpy() * self.const_factor
          val_loss += [curr_loss]

        val_loss = float(np.mean(np.concatenate(val_loss)))
        logger.info("Avg. Validation MAE = {}".format(val_loss))
        self.writer.add_scalar('val_loss', val_loss, iter_count)
        results['val_loss'] += [val_loss]

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
              model.module if self.use_gpu else model,
              optimizer,
              self.config,
              epoch + 1,
              tag='best')

        logger.info("Current Best Validation MAE = {}".format(best_val_loss))

        # check early stop
        if early_stop.tick([val_loss]):
          snapshot(
              model.module if self.use_gpu else model,
              optimizer,
              self.config,
              epoch + 1,
              tag='last')
          self.writer.close()
          break

      # training
      model.train()
      lr_scheduler.step()
      for data in train_loader:
        optimizer.zero_grad()

        if self.use_gpu:
          data['node_feat'], data['node_mask'], data['label'] = data_to_gpu(
              data['node_feat'], data['node_mask'], data['label'])

          if self.model_conf.name == 'LanczosNet':
            data['L'], data['D'], data['V'] = data_to_gpu(
                data['L'], data['D'], data['V'])
          elif self.model_conf.name == 'GraphSAGE':
            data['nn_idx'], data['nonempty_mask'] = data_to_gpu(
                data['nn_idx'], data['nonempty_mask'])
          elif self.model_conf.name == 'GPNN':
            data['L'], data['L_cluster'], data['L_cut'] = data_to_gpu(
                data['L'], data['L_cluster'], data['L_cut'])
          else:
            data['L'] = data_to_gpu(data['L'])[0]

        if self.model_conf.name == 'AdaLanczosNet':
          _, train_loss = model(
              data['node_feat'],
              data['L'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'LanczosNet':
          _, train_loss = model(
              data['node_feat'],
              data['L'],
              data['D'],
              data['V'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'GraphSAGE':
          _, train_loss = model(
              data['node_feat'],
              data['nn_idx'],
              data['nonempty_mask'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'GPNN':
          _, train_loss = model(
              data['node_feat'],
              data['L'],
              data['L_cluster'],
              data['L_cut'],
              label=data['label'],
              mask=data['node_mask'])
        else:
          _, train_loss = model(
              data['node_feat'],
              data['L'],
              label=data['label'],
              mask=data['node_mask'])

        # assign gradient
        train_loss.backward()
        optimizer.step()
        train_loss = float(train_loss.data.cpu().numpy())
        self.writer.add_scalar('train_loss', train_loss, iter_count)
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(
              epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module
                 if self.use_gpu else model, optimizer, self.config, epoch + 1)

    results['best_val_loss'] += [best_val_loss]
    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation MAE = {}".format(best_val_loss))

    return best_val_loss

  def test(self):
    test_dataset = eval(self.dataset_conf.loader_name)(
        self.config, split='test')
    # create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self.test_conf.batch_size,
        shuffle=False,
        num_workers=self.test_conf.num_workers,
        collate_fn=test_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)
    load_model(model, self.test_conf.test_model)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    model.eval()
    test_loss = []
    for data in tqdm(test_loader):
      if self.use_gpu:
        data['node_feat'], data['node_mask'], data['label'] = data_to_gpu(
            data['node_feat'], data['node_mask'], data['label'])

        if self.model_conf.name == 'LanczosNet':
          data['D'], data['V'] = data_to_gpu(data['D'], data['V'])
        elif self.model_conf.name == 'GraphSAGE':
          data['nn_idx'], data['nonempty_mask'] = data_to_gpu(
              data['nn_idx'], data['nonempty_mask'])
        elif self.model_conf.name == 'GPNN':
          data['L'], data['L_cluster'], data['L_cut'] = data_to_gpu(
              data['L'], data['L_cluster'], data['L_cut'])
        else:
          data['L'] = data_to_gpu(data['L'])[0]

      with torch.no_grad():
        if self.model_conf.name == 'AdaLanczosNet':
          pred, _ = model(
              data['node_feat'],
              data['L'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'LanczosNet':
          pred, _ = model(
              data['node_feat'],
              data['L'],
              data['D'],
              data['V'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'GraphSAGE':
          pred, _ = model(
              data['node_feat'],
              data['nn_idx'],
              data['nonempty_mask'],
              label=data['label'],
              mask=data['node_mask'])
        elif self.model_conf.name == 'GPNN':
          pred, _ = model(
              data['node_feat'],
              data['L'],
              data['L_cluster'],
              data['L_cut'],
              label=data['label'],
              mask=data['node_mask'])
        else:
          pred, _ = model(
              data['node_feat'],
              data['L'],
              label=data['label'],
              mask=data['node_mask'])

        curr_loss = (
            pred - data['label']).abs().cpu().numpy() * self.const_factor
        test_loss += [curr_loss]

    test_loss = float(np.mean(np.concatenate(test_loss)))
    logger.info("Test MAE = {}".format(test_loss))

    return test_loss
