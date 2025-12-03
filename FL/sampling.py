'''Some helper functions
'''
import random
from random import shuffle
random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
import tensorflow as tf 
import pandas as pd
from datasets import *
from collections import defaultdict

def distribute_dataset(dataset_name, num_peers, num_classes, dd_type = 'IID', classes_per_peer = 1, samples_per_class = 582, 
alpha = 1, attacker_source_class=None, attacker_target_class=None, num_attackers=0, seed= None):
    print("--> Loading of {} dataset".format(dataset_name))
    tokenizer = None
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist()
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    elif dataset_name == 'IMDB':
        trainset, testset, tokenizer = get_imdb(num_peers = num_peers)
    if dd_type == 'IID':
        peers_data_dict = split_iid_evenly(trainset, num_peers)
        attackers_list = []
    elif dd_type == 'MILD_NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
        attackers_list = []
    elif dd_type == 'Dirichlet':
        peers_data_dict = sample_dirichlet_fixed(
            dataset=trainset,
            num_users=num_peers,
            alpha=alpha,
            seed=seed
        )
        attackers_list = []
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict, attackers_list = sample_extreme_with_attackers(
            dataset=trainset,
            num_users=num_peers,
            num_classes=num_classes,
            classes_per_peer=classes_per_peer,
            samples_per_class=samples_per_class,
            attacker_source_class=attacker_source_class,
            attacker_target_class=attacker_target_class,
            num_attackers=num_attackers
        )
    else:
        raise ValueError(f"Unknown dd_type: {dd_type}")

    print("--> Dataset has been loaded!")
    return trainset, testset, peers_data_dict, tokenizer, attackers_list


def get_mnist():
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    trainset = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True,
                        transform=transform)
    return trainset, testset

def get_cifar10():
    data_dir = 'data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    testset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    return trainset, testset

def get_imdb(num_peers = 10):
    MAX_LEN = 128
    df = pd.read_csv('data/imdb.csv')
    df.sentiment = df.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

        
    train_df = df.iloc[:40000].reset_index(drop=True)
    valid_df = df.iloc[40000:].reset_index(drop=True)

    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    trainset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    # initialize dataset class for validation
    testset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)
   
    return trainset, testset, tokenizer


def sample_dirichlet(dataset, num_users, alpha=1):
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        if type(label) == torch.Tensor:
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())
    
    peers_data_dict = {i: {'data':np.array([]), 'labels':set()} for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        for user in range(num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)), axis=0)
            if num_imgs > 0:
                peers_data_dict[user]['labels'].add(n)

            classes[n] = classes[n][min(len(classes[n]), num_imgs):]
   
    return peers_data_dict


def sample_dirichlet_fixed(dataset, num_users, alpha=1.0, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    classes = defaultdict(list)
    for idx, (x, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        classes[label].append(idx)

    peers_data_dict = {i: {'data': [], 'labels': set()} for i in range(num_users)}

    for class_label, indices in classes.items():
        class_size = len(indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_users))
        sampled_nums = np.random.multinomial(class_size, proportions)

        start_idx = 0
        for user_id in range(num_users):
            end_idx = start_idx + sampled_nums[user_id]
            if sampled_nums[user_id] > 0:
                user_samples = indices[start_idx:end_idx]
                peers_data_dict[user_id]['data'].extend(user_samples)
                peers_data_dict[user_id]['labels'].add(class_label)
            start_idx = end_idx

    for user_id in range(num_users):
        peers_data_dict[user_id]['data'] = np.array(peers_data_dict[user_id]['data'], dtype=int)
        peers_data_dict[user_id]['labels'] = sorted(peers_data_dict[user_id]['labels'])

    return peers_data_dict


def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    n = len(dataset)
    num_classes = 10
    peers_data_dict = {i: {'data': np.array([]), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}
    for l in range(num_classes):
        label_idxs = np.where(labels == l)[0]
        label_indices[l] = list(idxs[label_idxs])

    available_labels = list(range(num_classes))

    for i in range(num_users):
        if len(available_labels) < classes_per_peer:
            raise ValueError(
                f"Not enough classes left to assign {classes_per_peer} classes per peer. "
                f"Only {len(available_labels)} classes remain: {available_labels}"
            )


        user_labels = np.random.choice(available_labels, classes_per_peer, replace=False).tolist()

        for l in user_labels:

            num_needed = min(samples_per_class, len(label_indices[l]))
            lab_idxs = label_indices[l][:num_needed]
            peers_data_dict[i]['data'] = np.concatenate((peers_data_dict[i]['data'], lab_idxs), axis=0)
            peers_data_dict[i]['labels'].append(l)

            label_indices[l] = label_indices[l][num_needed:]

            if len(label_indices[l]) < samples_per_class and l in available_labels:
                available_labels.remove(l)

    return peers_data_dict


def sample_extreme_with_attackers(
    dataset,
    num_users,
    num_classes=10,
    classes_per_peer=3,
    samples_per_class=100,
    attacker_source_class=None,
    attacker_target_class=None,
    num_attackers=0
):
    n = len(dataset)
    peers_data_dict = {i: {'data': np.array([], dtype=int), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    sorted_labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}
    for l in range(num_classes):
        class_mask = (sorted_labels == l)
        label_indices[l] = list(idxs[class_mask])

    attackers_list = []

    class_client_count = {l: 0 for l in range(num_classes)}

    max_clients_per_class = max(2, (num_users * classes_per_peer) // num_classes + 1)

    if num_attackers > 0:
        assert attacker_source_class is not None, "必须指定 attacker_source_class 才能分配攻击者"
        attackers_list = np.random.choice(num_users, num_attackers, replace=False).tolist()

        for cid in attackers_list:

            user_labels = [attacker_source_class]
            class_client_count[attacker_source_class] += 1

            other_candidates = [
                c for c in range(num_classes)
                if c != attacker_source_class and class_client_count[c] < max_clients_per_class
            ]
            num_others = classes_per_peer - 1
            if len(other_candidates) < num_others:

                other_candidates = [
                    c for c in range(num_classes)
                    if c != attacker_source_class
                ]
                other_candidates.sort(key=lambda x: class_client_count[x])

            additional_classes = []
            for _ in range(num_others):
                valid = [c for c in other_candidates if class_client_count[c] < max_clients_per_class]
                if not valid:
                    valid = other_candidates
                if valid:
                    chosen = valid[np.random.randint(len(valid))]
                    additional_classes.append(chosen)
                    class_client_count[chosen] += 1
                    if chosen in other_candidates:
                        other_candidates.remove(chosen)

            user_labels.extend(additional_classes)
            peers_data_dict[cid]['labels'] = user_labels.copy()

            for l in user_labels:
                num_needed = min(samples_per_class, len(label_indices[l]))
                if num_needed == 0:
                    raise ValueError(f"类别 {l} 无样本可分配给客户端 {cid}")
                lab_idxs = label_indices[l][:num_needed]
                peers_data_dict[cid]['data'] = np.concatenate([peers_data_dict[cid]['data'], lab_idxs])
                label_indices[l] = label_indices[l][num_needed:]

    covered_classes = set()
    for cid in range(num_users):
        covered_classes.update(peers_data_dict[cid]['labels'])
    uncovered_classes = set(range(num_classes)) - covered_classes
    normal_clients = [i for i in range(num_users) if i not in attackers_list]

    for cid in normal_clients:
        if not uncovered_classes:
            break
        must_include = uncovered_classes.pop()
        user_labels = [must_include]
        class_client_count[must_include] += 1

        other_candidates = [
            c for c in range(num_classes)
            if c != must_include and class_client_count[c] < max_clients_per_class
        ]
        if not other_candidates:
            other_candidates = [c for c in range(num_classes) if c != must_include]

        other_candidates.sort(key=lambda x: class_client_count[x])

        num_others = classes_per_peer - 1
        additional_classes = []
        for _ in range(num_others):
            valid = [c for c in other_candidates if class_client_count[c] < max_clients_per_class]
            if not valid:
                valid = other_candidates
            if valid:
                chosen = valid[np.random.randint(len(valid))]
                additional_classes.append(chosen)
                class_client_count[chosen] += 1
                if chosen in other_candidates:
                    other_candidates.remove(chosen)

        user_labels.extend(additional_classes)
        peers_data_dict[cid]['labels'] = user_labels.copy()

        for l in user_labels:
            num_needed = min(samples_per_class, len(label_indices[l]))
            if num_needed == 0:
                continue
            lab_idxs = label_indices[l][:num_needed]
            peers_data_dict[cid]['data'] = np.concatenate([peers_data_dict[cid]['data'], lab_idxs])
            label_indices[l] = label_indices[l][num_needed:]

    remaining_clients = [cid for cid in normal_clients if not peers_data_dict[cid]['data'].size]
    for cid in remaining_clients:
        candidate_pool = list(range(num_classes))
        candidate_pool.sort(key=lambda x: class_client_count[x])

        user_labels = []
        for _ in range(classes_per_peer):
            selected = None
            for c in candidate_pool:
                if c not in user_labels and class_client_count[c] < max_clients_per_class * 1.5:
                    selected = c
                    break
            if selected is None:

                selected = np.random.choice(candidate_pool)
            user_labels.append(selected)
            class_client_count[selected] += 1

        peers_data_dict[cid]['labels'] = user_labels.copy()
        for l in user_labels:
            num_needed = min(samples_per_class, len(label_indices[l]))
            if num_needed == 0:
                continue
            lab_idxs = label_indices[l][:num_needed]
            peers_data_dict[cid]['data'] = np.concatenate([peers_data_dict[cid]['data'], lab_idxs])
            label_indices[l] = label_indices[l][num_needed:]

    return peers_data_dict, attackers_list
def split_iid_evenly(dataset, num_peers=10):

    indices_per_class = [[] for _ in range(10)]
    for idx, target in enumerate(dataset.targets):
        indices_per_class[target].append(idx)

    peer_data_dict = {i: {'data': [], 'labels': []} for i in range(num_peers)}

    for cls in range(10):
        indices = np.array(indices_per_class[cls])
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_peers)

        for peer_id in range(num_peers):
            peer_indices = splits[peer_id].tolist()
            peer_data_dict[peer_id]['data'].extend(peer_indices)
    for peer_id in range(num_peers):
        data_indices = peer_data_dict[peer_id]['data']
        labels = [dataset.targets[i].item() for i in data_indices]
        peer_data_dict[peer_id]['labels'] = list(set(labels))

    return peer_data_dict