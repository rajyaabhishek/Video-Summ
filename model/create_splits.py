from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import math
import numpy as np
from utils_2 import write_json

args = { "dataset":r"C:\Users\abhis\OneDrive\Desktop\video summarization\CA-SUM-main\data\SumMe\eccv16_dataset_summe_google_pool5.h5",
"save_dir": r"C:\Users\abhis\OneDrive\Desktop\video summarization\CA-SUM-main\summaries",
"save_name": 'splits',
"num_splits": 5,
"train_percent": 0.8}


def split_random(keys, num_videos, num_train):
    """Random split"""
    train_keys, test_keys = [], []
    rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
    for key_idx, key in enumerate(keys):
        if key_idx in rnd_idxs:
            train_keys.append(key)
        else:
            test_keys.append(key)
    assert len(set(train_keys) & set(test_keys)) == 0, "Error: train_keys and test_keys overlap"
    return train_keys, test_keys


def create():
    print("==========\nArgs:{}\n==========".format(args))
    print("Goal: randomly split data for {} times, {:.1%} for training and the rest for testing".format(args["num_splits"], args["train_percent"]))
    print("Loading dataset from {}".format(args["dataset"]))
    dataset = h5py.File(args["dataset"], 'r')
    keys = dataset.keys()
    num_videos = len(keys)
    num_train = int(math.ceil(num_videos * args["train_percent"]))
    num_test = num_videos - num_train
    print("Split breakdown: # total videos {}. # train videos {}. # test videos {}".format(num_videos, num_train, num_test))
    splits = []
    for split_idx in range(args["num_splits"]):
        train_keys, test_keys = split_random(keys, num_videos, num_train)
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys,
            })
    saveto = osp.join(args["save_dir"], args["save_name"] + '.json')
    write_json(splits, saveto)
    print("Splits saved to {}".format(saveto))
    dataset.close()
    
if __name__ == '__main__':
    create()    