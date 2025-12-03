#!/usr/bin/env python
# coding: utf-8

#================================= Start of importing required packages and libraries =========================================#
from __future__ import print_function
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
import torch
from experiment_federated import *
import random
import threading

_malicious_baseline_record = {}
_malicious_baseline_lock = threading.Lock()



DATASET_NAME = "MNIST"
MODEL_NAME = "CNNMNIST"
DD_TYPE = 'IID'
ALPHA = 1
NUM_PEERS = 10
FRAC_PEERS = 1
SEED = 7
random.seed(SEED)
CRITERION = nn.CrossEntropyLoss()
GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 3
TEST_BATCH_SIZE = 1000
LOCAL_BS = 64
LOCAL_LR =  0.001
LOCAL_MOMENTUM = 0.9
NUM_CLASSES = 10
LABELS_DICT = {'Zero':0,
               'One':1,
               'Two':2,
               'Three':3,
               'Four':4,
               'Five':5,
               'Six':6,
               'Seven':7,
               'Eight':8,
               'Nine':9}

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = torch.device(DEVICE)
SOURCE_CLASS = 4
TARGET_CLASS = 9

CLASS_PER_PEER = 6
SAMPLES_PER_CLASS = 582
RATE_UNBALANCE = 1

start_defense_epoch = 1

ATTACK_TYPE = 'label_flipping2'
MALICIOUS_BEHAVIOR_RATE = 1


if __name__ == '__main__':

    attackers_ratios = [0.5]
    RULE = 'fedavg'

    log_filename = f"./results/{DATASET_NAME}_{MODEL_NAME}_{DD_TYPE}_{RULE}_summary_{SEED}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"联邦学习实验汇总报告\n")
        f.write(f"数据集: {DATASET_NAME}, 模型: {MODEL_NAME}, 数据分布: {DD_TYPE}\n")
        f.write(f"聚合规则: {RULE}, 随机种子: {SEED}\n")
        f.write(f"攻击类型: {ATTACK_TYPE}, 恶意行为率: {MALICIOUS_BEHAVIOR_RATE}\n")
        f.write(f"源类: {SOURCE_CLASS}, 目标类: {TARGET_CLASS}\n")
        f.write(f"全局轮数: {GLOBAL_ROUNDS}, 本地轮数: {LOCAL_EPOCHS}\n")
        f.write("="*80 + "\n")
        f.write(f"{'攻击比例':<10} {'最终全局准确率(%)':<20} {'源类准确率(%)':<20} {'平均CPU耗时(s)':<15}\n")
        f.write("-"*80 + "\n")

    print("Starting multi-round experiments with varying attackers_ratio...")
    
    for atr in attackers_ratios:
        print(f"\n{'='*60}")
        print(f"Running experiment with attackers_ratio = {atr}")
        print(f"{'='*60}")
        if RULE == 'ShieldFL':
            aggregator = ShieldFL_GradientAggregator(start_defense_epoch)
        elif RULE == 'similarityweightedavg':
            aggregator = SimilarityWeightedAvg(start_defense_epoch, atr)

        run_exp(
            dataset_name=DATASET_NAME,
            model_name=MODEL_NAME,
            dd_type=DD_TYPE,
            num_peers=NUM_PEERS,
            frac_peers=FRAC_PEERS,
            seed=SEED,
            test_batch_size=TEST_BATCH_SIZE,
            criterion=CRITERION,
            global_rounds=GLOBAL_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            local_bs=LOCAL_BS,
            local_lr=LOCAL_LR,
            local_momentum=LOCAL_MOMENTUM,
            labels_dict=LABELS_DICT,
            device=DEVICE,
            attackers_ratio=atr, 
            attack_type=ATTACK_TYPE,
            malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
            rule=RULE,
            source_class=SOURCE_CLASS,
            target_class=TARGET_CLASS,
            class_per_peer=CLASS_PER_PEER,
            samples_per_class=SAMPLES_PER_CLASS,
            rate_unbalance=RATE_UNBALANCE,
            alpha=ALPHA,
            resume=False,
            #aggregator=aggregator
        )

        print(f"Experiment with attackers_ratio={atr} completed.\n")