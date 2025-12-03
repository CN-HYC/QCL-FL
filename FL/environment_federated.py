from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
from datasets import *
import os
import random
import threading
from tqdm import tqdm as tqdm_notebook
import copy
from operator import itemgetter
import time
from aggregation import *
from torch.utils.data import Subset
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if ' _malicious_baseline_record' not in globals():
    _malicious_baseline_record = {}
    _malicious_baseline_lock = threading.Lock()

    class MultiSourcePoisonedDataset(data.Dataset):
        def __init__(self, dataset, source_classes=None, target_classes=None):

            self.dataset = dataset
            self.source_classes = source_classes if source_classes is not None else []
            self.target_classes = target_classes if target_classes is not None else []

        def __getitem__(self, index):
            x, y = self.dataset[index][0], self.dataset[index][1]
            original_label = y
            new_label = y
            flipped = False

            for s, t in zip(self.source_classes, self.target_classes):
                if y == s:
                    new_label = t
                    flipped = True
                    break
            return x, new_label

        def __len__(self):
            return len(self.dataset)

class LabelF:
    @staticmethod
    def targeted_flip(dataset, source_class, target_class):
        if not isinstance(dataset, torch.utils.data.TensorDataset):
            raise TypeError("Only supports TensorDataset")
        x, y = dataset.tensors
        y_new = y.clone()
        y_new[y == source_class] = target_class
        return torch.utils.data.TensorDataset(x, y_new)

    @staticmethod
    def random_flip(dataset, num_classes, flip_ratio=0.2):
        if not isinstance(dataset, torch.utils.data.TensorDataset):
            raise TypeError("Only supports TensorDataset")
        x, y = dataset.tensors
        y_new = y.clone()
        flip_mask = (torch.rand(len(y), device=y.device) < flip_ratio)
        
        if flip_mask.any():
            new_targets = torch.randint(0, num_classes - 1, size=(flip_mask.sum(),), device=y.device)
            new_targets += (new_targets >= y[flip_mask]).int()
            new_targets %= num_classes
            y_new[flip_mask] = new_targets

        return torch.utils.data.TensorDataset(x, y_new)
        
class Peer():
    _performed_attacks = 0

    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self, val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion,
                 device, local_epochs, local_bs, local_lr,
                 local_momentum, peer_type='honest',source_class=None, target_class=None, num_classes=None):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
        self.source_class = source_class
        self.target_class = target_class
        self.num_classes = num_classes
    # -----------------------------------------------------------------------

    def participant_update(self, global_epoch, model, attack_type='no_attack', malicious_behavior_rate=0,
                       source_class=None, target_class=None, dataset_name=None):
        device = next(model.parameters()).device
        epochs = self.local_epochs
        attacked = 0

        if not hasattr(self, '_gpu_data'):
            if isinstance(self.local_data, torch.utils.data.TensorDataset):
                x, y = self.local_data.tensors
                self._gpu_data = torch.utils.data.TensorDataset(x.to(device), y.to(device))
            else:
                self._gpu_data = self.local_data  # fallback

        do_attack = False
        if (self.peer_type == 'attacker') and (global_epoch >= 0):
            if np.random.random() <= malicious_behavior_rate:
                do_attack = True
                attacked = 1
                self.performed_attacks += 1

        current_data = self._gpu_data
        if do_attack:
            if attack_type == 'label_flipping' and dataset_name != 'IMDB':
                current_data = LabelF.targeted_flip(current_data, source_class, target_class)
                print(f" [Attack] Peer {self.peer_id} LABEL FLIPPING ({source_class}→{target_class})")
            elif attack_type == 'label_flipping2' and dataset_name != 'IMDB':
                current_data = LabelF.random_flip(current_data, self.num_classes, flip_ratio=malicious_behavior_rate)
                print(f" [Attack] Peer {self.peer_id} RANDOM LABEL FLIP")

        train_loader = DataLoader(
            current_data,
            batch_size=self.local_bs,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.local_lr,
                                  momentum=self.local_momentum, weight_decay=5e-4)

        # ===== Training Loop =====
        start_train = time.time()
        epochs_loss = []
        t = 0
        for epoch in range(epochs):
            epoch_loss = []
            for data, target in train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                cur_time = time.time()
                optimizer.step()
                t += time.time() - cur_time
                epoch_loss.append(loss.item())
            epochs_loss.append(np.mean(epoch_loss))
        local_train_time = time.time() - start_train

        if attack_type == 'gaussian' and self.peer_type == 'attacker' and global_epoch >= 0:
            with torch.no_grad():
                for param in model.parameters():
                    if np.random.random() < malicious_behavior_rate:
                        param.add_(torch.randn_like(param) * 0.1)
            print(f"[Gaussian Attack] Peer {self.peer_id} added noise!")
            attacked = 1
            self.performed_attacks += 1

        with torch.no_grad():
            gpu_weights = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
            }

        avg_loss = np.mean(epochs_loss) if epochs_loss else 0.0
        return gpu_weights, None, avg_loss, attacked, t
    def participant_update_mod(self, global_epoch, global_model, attack_type='no_attack',
                               malicious_behavior_rate=0, source_classes=None, target_classes=None,
                               dataset_name=None):
        START_ATTACK_EPOCH = 0
        epochs = self.local_epochs
        attacked = 0

        if not hasattr(self, '_gpu_data'):
            if isinstance(self.local_data, torch.utils.data.TensorDataset):
                x, y = self.local_data.tensors
                self._gpu_data = torch.utils.data.TensorDataset(x.to(self.device), y.to(self.device))
            else:
                self._gpu_data = self.local_data  # fallback
        current_data = self._gpu_data

        do_attack = False
        if self.peer_type == 'attacker' and global_epoch >= START_ATTACK_EPOCH:
            if np.random.random() <= malicious_behavior_rate:
                do_attack = True
                attacked = 1
                self.performed_attacks += 1

        if do_attack:
            if attack_type == 'label_flipping' and dataset_name != 'IMDB':
                current_data = LabelF.targeted_flip(current_data, source_classes, target_classes)
                print(f" [Attack] Peer {self.peer_id} LABEL FLIPPING ({source_classes}→{target_classes})")
            elif attack_type == 'label_flipping2' and dataset_name != 'IMDB':
                current_data = LabelF.random_flip(current_data, self.num_classes, flip_ratio=malicious_behavior_rate)
                print(f" [Attack] Peer {self.peer_id} RANDOM LABEL FLIP")
        train_loader = DataLoader(
            current_data,
            batch_size=self.local_bs,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        fc2_weight_shape = None
        for name, param in model.named_parameters():
            if name == 'fc2.weight':
                fc2_weight_shape = param.shape
                break
        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=self.local_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr,
                                  momentum=self.local_momentum, weight_decay=5e-4)

        total_grad_sum = None
        clf_grad_sum = None
        total_batches = 0
        epochs_loss = []
        train_start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                current_grad = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

                fc2_grad = None
                for name, p in model.named_parameters():
                    if name == 'fc2.weight' and p.grad is not None:
                        fc2_grad = p.grad.detach().clone()
                        break

                if total_grad_sum is None:
                    total_grad_sum = current_grad
                else:
                    for i in range(len(total_grad_sum)):
                        total_grad_sum[i] += current_grad[i]

                if fc2_grad is not None:
                    if clf_grad_sum is None:
                        clf_grad_sum = fc2_grad
                    else:
                        clf_grad_sum += fc2_grad

                epoch_loss.append(loss.item())
                total_batches += 1

            if epoch_loss:
                epochs_loss.append(np.mean(epoch_loss))

        if total_grad_sum is None:
            return None, None, 0.0, attacked

        avg_grad = [g / total_batches for g in total_grad_sum]
        clf_grad_avg = (clf_grad_sum / total_batches) if clf_grad_sum is not None else \
                       torch.zeros(fc2_weight_shape, device=self.device)
        train_time = time.time() - train_start_time
        print(f"[Timer] Peer {self.peer_id} 本地训练总耗时: {train_time:.4f} 秒")

        if (self.peer_type == 'attacker' and
            global_epoch >= START_ATTACK_EPOCH and
            attack_type == 'gaussian' and
            np.random.random() <= malicious_behavior_rate):

            scale = 1.0
            with torch.no_grad():
                for i in range(len(avg_grad)):
                    if avg_grad[i] is not None:
                        noise = torch.randn_like(avg_grad[i]) * scale
                        avg_grad[i] += noise
                noise_clf = torch.randn_like(clf_grad_avg) * scale
                clf_grad_avg += noise_clf
            attacked = 1
            self.performed_attacks += 1

        with torch.no_grad():
            raw_grad_flat = torch.cat([g.flatten() for g in avg_grad], dim=0)
            clf_grad_flat = clf_grad_avg.flatten()

        avg_loss = np.mean(epochs_loss) if epochs_loss else 0.0

        return raw_grad_flat, clf_grad_flat, avg_loss, attacked

class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers,
                 seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
                 local_momentum, labels_dict, device, attackers_ratio=0,
                 class_per_peer=2, samples_per_class=250, rate_unbalance=1, alpha=1, source_class=None, target_class=None, num_attackers=0,
                 aggregator=None):

        FL._history = np.zeros(num_peers)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i + 1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer 
        self.class_ = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.trainset, self.testset = None, None
        self.target_class = target_class
        self.num_attackers = num_attackers
        self.aggregator = aggregator
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        self.trainset, self.testset, user_groups_train, tokenizer, self.attackers_list = distribute_dataset(
            self.dataset_name,
            self.num_peers,
            self.num_classes,
            self.dd_type,
            self.class_per_peer,
            self.samples_per_class,
            self.alpha,
            attacker_source_class=self.source_class,
            attacker_target_class=self.target_class,
            num_attackers=self.num_attackers
        )
        self.test_loader = DataLoader(self.testset, batch_size=self.test_batch_size,
                                      shuffle=False, num_workers=1)
        self.clean_samples = 100
        self.clean_batch_size = 64

        if hasattr(self.testset, 'targets'):
            targets = np.array(self.trainset.targets)
        elif hasattr(self.testset, 'labels'):
            targets = np.array(self.trainset.labels)
        else:
            raise ValueError("Test dataset must have 'targets' or 'labels' attribute")

        num_classes = len(np.unique(targets))
        rng = np.random.RandomState(self.seed)


        samples_per_class = self.clean_samples // num_classes
        extra = self.clean_samples % num_classes

        clean_indices = []

        for cls in range(num_classes):

            class_indices = np.where(targets == cls)[0]
            n_sample = samples_per_class + (1 if extra > 0 else 0)
            n_sample = min(n_sample, len(class_indices))

            sampled = rng.choice(class_indices, size=n_sample, replace=False)
            clean_indices.extend(sampled.tolist())

            if extra > 0:
                extra -= 1

        clean_indices = np.array(clean_indices)
        rng.shuffle(clean_indices)

        clean_subset = Subset(self.trainset, clean_indices)
        self.clean_loader = DataLoader(clean_subset, batch_size=self.clean_batch_size, shuffle=False)

        print(f" Created balanced clean_loader from Train set: {len(clean_subset)} samples "
              f"(~{self.clean_samples // num_classes} per class), batch_size={self.clean_batch_size}")

        unique, counts = np.unique(targets[clean_indices], return_counts=True)
        print(" Clean loader class distribution:", dict(zip(unique, counts)))
        
        
        

        self.global_model = setup_model(model_architecture=self.model_name, num_classes=self.num_classes,
                                        tokenizer=tokenizer, embedding_dim=self.embedding_dim)
        self.global_model = self.global_model.to(self.device)

        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            indices = user_groups_train[p]['data']
            indices = [int(i) for i in indices]

            x = []
            y = []
            for idx in indices:
                data, target = self.trainset[int(idx)]
                x.append(data)
                y.append(target)
            x = torch.stack(x)
            y = torch.tensor(y)

            peer_data = torch.utils.data.TensorDataset(x, y)

            self.local_data.append(peer_data)
            if self.source_class in user_groups_train[p]['labels']:
                self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        print('--> Creating peers instances')

        final_attackers = []

        if len(self.attackers_list) > 0:

            for aid in self.attackers_list:
                if aid not in self.have_source_class:
                    print(
                        f"Attacker {aid} has no source class {self.source_class} (may reduce attack effectiveness)")

            final_attackers = self.attackers_list
        else:
            print("⚠️  attackers_list is empty. Using ratio-based attacker selection.")

            k_src = len(self.have_source_class)
            print(f'# of peers who have source class examples: {k_src}')

            m_ = int(self.num_attackers)
            m_ = min(m_, k_src)

            if m_ <= 0:
                print("No attackers created: not enough clients with source class.")
            else:
                print(f"Selecting {m_} attackers from clients with source class...")
                candidates = self.have_source_class.copy()
                random.shuffle(candidates)
                final_attackers = candidates[:m_]
                print(f"Randomly selected attackers: {sorted(final_attackers)}")
        peers = list(range(self.num_peers))

        for i in peers:
            if i in final_attackers:
                self.peers.append(Peer(
                    peer_id=i,
                    peer_pseudonym=self.peers_pseudonyms[i],
                    local_data=self.local_data[i],
                    labels=self.labels[i],
                    criterion=self.criterion,
                    device=self.device,
                    local_epochs=self.local_epochs,
                    local_bs=self.local_bs,
                    local_lr=self.local_lr,
                    local_momentum=self.local_momentum,
                    peer_type='attacker',
                    source_class=self.source_class,
                    target_class=self.target_class,
                    num_classes=self.num_classes,
                ))
            else:
                self.peers.append(Peer(
                    peer_id=i,
                    peer_pseudonym=self.peers_pseudonyms[i],
                    local_data=self.local_data[i],
                    labels=self.labels[i],
                    criterion=self.criterion,
                    device=self.device,
                    local_epochs=self.local_epochs,
                    local_bs=self.local_bs,
                    local_lr=self.local_lr,
                    local_momentum=self.local_momentum,
                    peer_type='honest'
                ))

        self.final_attackers = final_attackers
        del self.local_data
        

    def test(self, model, device, test_loader, dataset_name=None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if dataset_name == 'IMDB':
                test_loss.append(self.criterion(output, target.view(-1, 1)).item())
                pred = output > 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss.append(self.criterion(output, target).item())
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            n += target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n, 100 * correct / n))
        return 100.0 * (float(correct) / n), test_loss

    def test_label_predictions(self, model, device, test_loader, dataset_name=None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)

                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]

    def choose_peers(self):
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)
        return selected_peers
    
    

    def run_experiment(self, aggregator, attack_type='no_attack', malicious_behavior_rate=0,
                       source_class=None, target_class=None, rule='fedavg', resume=False):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        fg = FoolsGoldmod(
            num_peers=self.num_peers,
            clf_layer_name='fc2.weight',
            device=self.device,
            model_fn=lambda: create_model(self.model_name, self.num_classes)
        )
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        if aggregator is not None:
            self.aggregator = aggregator
        elif resume and not hasattr(self, 'aggregator'):
            raise ValueError("Attempting to resume training without providing an aggregator.")


        start_round = 0

        f = max(1, int(self.attackers_ratio * self.num_peers))
        self.krum_aggregator = Krum_Aggregator(
            f=f,
            multi_krum=True,
            start_decay_epoch=0,
            total_rounds=self.global_rounds
        )

        self.trimmed_mean_aggregator = TrimmedMean_Aggregator(
            trim_ratio=self.attackers_ratio,
            start_decay_epoch=0,
            total_rounds=self.global_rounds
        )

        self.fedavg_aggregator = FedAvg_Aggregator

        fltrust_aggregator = FLTrust_Aggregator(
            server_clean_loader=self.clean_loader,
            model_fn=lambda: create_model(self.model_name, self.num_classes),
            device=self.device,
            clf_layer_name="fc2.weight",
            use_server_anchor=False
        )
        self.rppd_aggregator = RPPD(
            start_defense_epoch=5,
            attackers_ratio=self.attackers_ratio,
            initial_lr=0.1,
            min_lr=0.01,
            total_rounds=self.global_rounds,
            clf_layer_name="fc2.weight"
        )
        
        
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load(
                './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
                    self.attackers_ratio) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']

            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")
        f = max(1, int(self.attackers_ratio * self.num_peers))

        # 创建聚合器实例
        krum_aggregator = Krum_Aggregator(
            f=f,
            multi_krum=True,
            start_decay_epoch=0,
            total_rounds=self.global_rounds
        )

        trimmed_mean_aggregator = TrimmedMean_Aggregator(
            trim_ratio=0.3,
            start_decay_epoch=0,
            total_rounds=self.global_rounds
        )

        fedavg_aggregator = FedAvg_Aggregator()
        
        fltrust_aggregator = FLTrust_Aggregator(
            server_clean_loader=self.clean_loader,
            model_fn=lambda: create_model(self.model_name, self.num_classes),
            device=self.device,
            clf_layer_name='fc2.weight',
        )
        shieldfl_mod_aggregator = ShieldFL_mod(
            device=self.device,
            clf_layer_name='fc2.weight'
        )
        
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()


            print(f'\n | Global training round : {epoch + 1}/{self.global_rounds} |\n')

            needs_models = rule in ['mkrum','foolsgold','RPPD','shieldflmod']
            needs_gradients = rule in ['similarityweightedavg','ShieldFL']
            selected_peers = self.choose_peers()

            local_weights, local_grads, local_models, local_losses, performed_attacks = [], [], [], [], []
            local_clf_grads = []
            peers_types = []
            i = 1
            attacks = 0
            Peer._performed_attacks = 0
            for peer in selected_peers:
                peers_types.append(self.peers[peer].peer_type)
                if needs_gradients and not needs_models:

                    full_grad, clf_grad, peer_loss, attacked = self.peers[peer].participant_update_mod(
                        global_epoch=epoch,
                        global_model=simulation_model,
                        attack_type=attack_type,
                        malicious_behavior_rate=malicious_behavior_rate,
                        source_classes=self.source_class,
                        target_classes=self.target_class,
                        dataset_name=self.dataset_name
                    )
                    local_grads.append(full_grad)
                    local_clf_grads.append(clf_grad)
                    local_losses.append(peer_loss)
                    attacks += attacked
                else:

                    peer_update, peer_local_model, peer_loss, attacked, runtime = self.peers[peer].participant_update(
                        epoch,
                        copy.deepcopy(simulation_model),
                        attack_type=attack_type,
                        malicious_behavior_rate=malicious_behavior_rate,
                        source_class=source_class,
                        target_class=target_class,
                        dataset_name=self.dataset_name
                    )
                    local_weights.append(peer_update)
                    local_models.append(peer_local_model)
                    local_losses.append(peer_loss)
                    attacks += attacked

            loss_avg = sum(local_losses) / len(local_losses)
            print('Average of peers\' local losses: {:.6f}'.format(loss_avg))

            scores = np.zeros(len(local_weights))

            if rule == 'fedavg_old':
                start_agg = time.time()
                global_weights = average_weights(local_weights, [1 for _ in range(len(local_weights))])
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'median':
                start_agg = time.time()
                global_weights = simple_median(local_weights)
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'rmedian':
                start_agg = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'tmean_old':
                start_agg = time.time()
                global_weights = trimmed_mean(local_weights, trim_ratio=self.attackers_ratio)
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'tmean':
                start_agg = time.time()
                global_weights = trimmed_mean_aggregator.aggregate(
                    global_model=self.global_model,
                    client_models=local_weights
                )
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'RPPD':
                print(f"[Round {epoch + 1}] Using RPPD aggregation.")
                start_agg = time.time()
                global_weights = self.rppd_aggregator.aggregate(
                    global_model=simulation_model.state_dict(),
                    client_models=local_weights,
                )
                cpu_runtimes.append(time.time() - start_agg)

                simulation_model.load_state_dict(global_weights)
                for client in self.peers:
                    if hasattr(client, 'model'):
                        client.model.load_state_dict(simulation_model.state_dict())

            elif rule == 'fedavg':
                start_agg = time.time()
                global_weights = fedavg_aggregator.aggregate(
                    global_model=self.global_model,
                    client_models=local_weights
                )
                agg_time = time.time() - start_agg
                cpu_runtimes.append(time.time() - start_agg)
                print(f"[Aggregation] Round {epoch + 1}: {agg_time:.4f} seconds")  # ← 新增

            elif rule == 'krum_old':
                start_agg = time.time()
                global_weights = krum_aggregator.aggregate(
                    global_model=self.global_model,
                    client_models=local_weights
                )
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'mkrum':
                print(f"[Round {epoch}] Using MKRUM aggregation with f={f}")
                start_agg = time.time()
                global_weights = krum_aggregator.aggregate(
                    global_model=simulation_model,
                    client_models=local_weights,
                )
                agg_time = time.time() - start_agg
                cpu_runtimes.append(time.time() - start_agg)
                print(f"[Aggregation] Round {epoch + 1}: {agg_time:.4f} seconds")  # ← 新增

            elif rule == 'foolsgold':
                print(f"[Round {epoch + 1}] Using Foolsgold (GPU)")
                start_agg = time.time()
                global_weights = fg.aggregate(
                    global_model=simulation_model.state_dict(),
                    client_models=local_weights,
                    selected_peers=selected_peers
                )
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'fltrust':
                print(f"[Round {epoch + 1}] Using FLTrust aggregation.")
                start_agg = time.time()
                global_weights = fltrust_aggregator.aggregate(
                    global_model=simulation_model.state_dict(),
                    client_models=local_weights
                )
                cpu_runtimes.append(time.time() - start_agg)
                simulation_model.load_state_dict(global_weights)

            elif rule == 'similarityweightedavg':
                print("[DEBUG] Using similarity-weighted aggregation.")
                start_agg = time.time()
                if hasattr(self, 'aggregator') and self.aggregator is not None:
                    global_weights = self.aggregator.aggregate(
                        global_model=copy.deepcopy(simulation_model),
                        client_gradients=local_grads,
                    )
                else:
                    global_weights = average_weights(local_weights)
                cpu_runtimes.append(time.time() - start_agg)

            elif rule == 'ShieldFL':
                print(f"[Round {epoch}] Using ShieldFL Gradient Aggregation.")
                start_agg = time.time()
                if hasattr(self, 'aggregator') and self.aggregator is not None:
                    global_weights = self.aggregator.aggregate(
                        global_model=simulation_model,
                        client_gradients=local_grads,
                        client_clf_gradients=local_clf_grads
                    )

                    if isinstance(global_weights, dict):
                        simulation_model.load_state_dict(global_weights)
                    else:
                        temp_optimizer = optim.Adam(simulation_model.parameters(), lr=self.local_lr)
                        for param, grad in zip(simulation_model.parameters(), global_weights):
                            if param.requires_grad:
                                param.grad = grad
                        temp_optimizer.step()

                    for client in self.peers:
                        if hasattr(client, 'model'):
                            client.model.load_state_dict(simulation_model.state_dict())
                else:
                    print("[Warning] ShieldFL aggregator not found. Falling back to FedAvg.")
                    global_weights = average_weights([model.state_dict() for model in local_models])
                    simulation_model.load_state_dict(global_weights)
                    for client in self.peers:
                        if hasattr(client, 'model'):
                            client.model.load_state_dict(simulation_model.state_dict())
                agg_time = time.time() - start_agg
                cpu_runtimes.append(time.time() - start_agg)
                print(f"[Aggregation] Round {epoch + 1}: {agg_time:.4f} seconds")  # ← 新增
                
            elif rule == 'shieldflmod':
                print(f"[Round {epoch + 1}] Using ShieldFL_mod (Modified Version) aggregation.")

                global_weights = shieldfl_mod_aggregator.aggregate(
                    global_model=simulation_model.state_dict(),
                    client_models=local_weights,
                )

                simulation_model.load_state_dict(global_weights)

                for client in self.peers:
                    if hasattr(client, 'model'):
                        client.model.load_state_dict(simulation_model.state_dict())
            else:
                start_agg = time.time()
                global_weights = average_weights(local_weights)

            if rule not in ['RPPD', 'fltrust', 'ShieldFL']:
                simulation_model.load_state_dict(global_weights)
                for client in self.peers:
                    if hasattr(client, 'model'):
                        try:
                            client.model.load_state_dict(simulation_model.state_dict())
                        except Exception as e:
                            print(f"[Error] Failed to update client model: {e}")

            g_model = copy.deepcopy(simulation_model)
            if epoch >= self.global_rounds - 10:
                last10_updates.append(global_weights)

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader,
                                                    dataset_name=self.dataset_name)
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks)

            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model': g_model,
                'local_models': copy.deepcopy(local_models),
                'last10_updates': last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
            }
            savepath = './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
                self.attackers_ratio) + '.t7'
            torch.save(state, savepath)

            del local_models, local_weights, local_grads
            gc.collect()
            torch.cuda.empty_cache()
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader,
                                                               dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class', 'Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i] / np.sum(r) * 100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i] / np.sum(r) * 100, 2))

            state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'avg_cpu_runtime': np.mean(cpu_runtimes)
            }
            savepath = './results/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
                self.attackers_ratio) + '.t7'
            print('Global accuracies: ', global_accuracies)
            print('Class {} accuracies: '.format(source_class), source_class_accuracies)
            print('Test loss:', test_losses)
            print('Average CPU aggregation runtime:', np.mean(cpu_runtimes))
            if epoch + 1 == 100:  # 最后一轮（epoch 从0开始，所以+1）
                final_log_path = f'./results/{self.dataset_name}_{self.model_name}_{self.dd_type}_{rule}_attackers{self.attackers_ratio}_final_rounds.txt'
                with open(final_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== 联邦学习实验完整轮次记录 ===\n")
                    f.write(f"数据集: {self.dataset_name}\n")
                    f.write(f"模型: {self.model_name}\n")
                    f.write(f"数据分布: {self.dd_type}\n")
                    f.write(f"聚合规则: {rule}\n")
                    f.write(f"攻击者比例: {self.attackers_ratio}\n")
                    f.write(f"源类别: {source_class}\n")
                    f.write(f"总轮数: {self.global_rounds}\n")
                    f.write(f"最终全局准确率: {current_accuracy:.2f}%\n")
                    f.write(f"最终源类准确率: {source_class_accuracies[-1] if len(source_class_accuracies) > 0 else 0:.2f}%\n")
                    f.write(f"平均CPU聚合耗时: {np.mean(cpu_runtimes):.6f}s\n")
                    f.write("="*80 + "\n\n")

                    f.write("Global Accuracies per Round (1~{}):\n".format(self.global_rounds))
                    f.write(str(global_accuracies))
                    f.write("\n\n" + "-"*80 + "\n\n")

                    f.write(f"Class {source_class} Accuracies per Round (1~{self.global_rounds}):\n")
                    f.write(str(source_class_accuracies))
                    f.write("\n\n" + "-"*80 + "\n\n")

                    f.write(" Test Losses per Round (1~{}):\n".format(self.global_rounds))
                    f.write(str(test_losses))
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("记录完成于第 {} 轮\n".format(epoch + 1))
