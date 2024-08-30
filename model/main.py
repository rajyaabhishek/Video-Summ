# -*- coding: utf-8 -*-
from __future__ import print_function
from configs import get_config
from solver import Solver
from data_loader import get_loader
import os
from utils_2 import read_json,write_json
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
import numpy as np  
from compute_reward import compute_reward


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(f"[Current split: {config.split_index}]: block_size={config.block_size} and "
          f"\u03C3={config.reg_factor} for {config.video_type} dataset.")
    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
    solver.train()
# tensorboard --logdir '.../CA-SUM/Summaries/' --host localhost



dataset = h5py.File(config.dataset, 'r')
num_videos = len(dataset.keys())
splits = read_json(r"C:\Users\abhis\OneDrive\Desktop\video summarization\CA-SUM-main\data\splits\summe_splits.json")
    
assert config.split_index < len(splits), "split_id (got {}) exceeds {}".format(config.split_index, len(splits))
split = splits[config.split_index]
train_keys = split['train_keys']
test_keys = split['test_keys']
start_epoch=0


baselines = {key: 0. for key in train_keys} # baseline rewards for videos
reward_writers = {key: [] for key in train_keys} # record reward changes for each video
for epoch in range(start_epoch, config.n_epochs):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) # shuffle indices
        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq) # input shape (seq_len, dim)  
            #
            trained_model = solver.model
            # input=torch.randn(100,1024)
            y,attn_weights = trained_model(seq) # output shape (1, seq_len, 1)
            probs = y.squeeze()
            #*
            cost = config.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(config.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                #* 
                use_gpu=False
                reward = compute_reward(seq, actions, use_gpu=use_gpu)
                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())
            optimizer = torch.optim.Adam(trained_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)    
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))
        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, config.n_epochs, epoch_reward))
write_json(reward_writers, osp.join(config.save_dir, 'rewards.json'))
solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
solver.train()
  
dataset = h5py.File(config.dataset, 'r')
num_videos = len(dataset.keys())
splits = read_json(r"C:\Users\abhis\OneDrive\Desktop\video summarization\CA-SUM-main\data\splits\summe_splits.json")
    
assert config.split_index < len(splits), "split_id (got {}) exceeds {}".format(config.split_index, len(splits))
split = splits[config.split_index]
train_keys = split['train_keys']
test_keys = split['test_keys']
start_epoch=0

             


#changes 
# seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1,seq_len, dim) TO  seq = torch.from_numpy(seq) # input shape (seq_len, dim)  
# to fit in the input shape of CASUM model 