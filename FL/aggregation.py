import copy
import enum
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from collections import OrderedDict
from utils import *
def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts

def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med

def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers

    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers] += grads
        wv = foolsgold(self.memory)
        self.wv_history.append(wv)
        return wv[selectec_peers]

def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med


def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])

    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


def average_weights(w, marks):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / sum(marks))
    return w_avg


def Krum(updates, f, multi=False):
    for i, update in enumerate(updates):
        if not isinstance(update, torch.nn.Module):
            raise TypeError(f"Krum: updates[{i}] is not a nn.Module, got {type(update)}")
        if not hasattr(update, 'parameters'):
            raise TypeError(f"Krum: updates[{i}] has no .parameters()")

    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.stack(updates)
    dist_matrix = torch.cdist(updates_, updates_, p=2) ** 2  # [n, n]
    k = n - f - 2
    if k <= 0:
        raise ValueError(f"k = n - f - 2 = {k} <= 0, not valid. Require n > 2f + 2.")

    scores = []
    for i in range(n):
        dist_i = dist_matrix[i]
        dist_i[i] = float('inf')
        nearest_k_distances, _ = torch.topk(dist_i, k, largest=False)
        score = nearest_k_distances.sum()
        scores.append(score)

    sorted_indices = torch.argsort(torch.tensor(scores))

    if multi:
        return sorted_indices[:n - f].tolist()
    else:
        return sorted_indices[0].item()

class SimilarityWeightedAvg:
    def __init__(self, start_defense_epoch=5, attackers_ratio=0.1,
                 initial_lr=0.5, min_lr=0.05, total_rounds=100, clf_layer_name=None):
        self.global_clf_grad_direction = None
        self.round = 0
        self.attackers_ratio = attackers_ratio
        self.start_defense_epoch = start_defense_epoch
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_rounds = total_rounds
        self.clf_layer_name = "fc2.weight"
        self.clf_slice = None

    def _normalize(self, vec):
        norm = vec.norm(p=2)
        return vec / (norm + 1e-8)

    def _get_clf_slice(self, model, clf_layer_name):
        if clf_layer_name is None:
            clf_layer_name = self.clf_layer_name
        idx = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            if name == clf_layer_name:
                start = idx
                end = idx + num_params
                return slice(start, end)
            idx += num_params
        raise ValueError(f"Classifier layer '{clf_layer_name}' not found in model.")

    def aggregate(self, global_model, client_models=None, client_gradients=None, client_clf_gradients=None):
        if client_gradients is None or len(client_gradients) == 0:
            return global_model.state_dict()

        device = next(global_model.parameters()).device
        n_clients = len(client_gradients)
        alpha = int(self.attackers_ratio * n_clients)
        if alpha >= n_clients:
            raise ValueError(f"Too many attackers: alpha={alpha}, n_clients={n_clients}")

        flat_grads = [g.to(device) for g in client_gradients]
        if self.clf_slice is None:
            self.clf_slice = self._get_clf_slice(global_model, self.clf_layer_name)
            print(f"[ClassifierOnly] Using classifier layer: {self.clf_layer_name}, slice: {self.clf_slice}")

        clf_grads = [g[self.clf_slice] for g in flat_grads]
        normalized_clf_grads = [self._normalize(g) for g in clf_grads]

        if self.round == 0 or self.round < self.start_defense_epoch or self.global_clf_grad_direction is None:
            print(f"[SimilarityWeightedAvg Round {self.round}] Warm-up phase: using simple average to initialize direction.")
            avg_clf_grad = torch.stack(clf_grads).mean(dim=0)
            self.global_clf_grad_direction = self._normalize(avg_clf_grad)

            avg_grad = torch.stack(flat_grads).mean(dim=0)
            aggregated_flat_grad = avg_grad
        else:
            current_clf_directions = torch.stack(normalized_clf_grads)  # [n_clients, D_clf]
            sims = torch.mv(current_clf_directions, self.global_clf_grad_direction)  # [n_clients]
            sims_np = sims.cpu().numpy()
            print(f"[SimilarityWeightedAvg Round {self.round}] ClfGrad Cosine similarities: {sims_np.round(4)}")

            _, indices_to_keep = torch.topk(sims, n_clients - alpha, largest=True)
            exclude_indices = [i for i in range(n_clients) if i not in indices_to_keep]
            print(f"[SimilarityWeightedAvg] Excluding clients: {exclude_indices}")
            selected_flat_grads = torch.stack(flat_grads)[indices_to_keep]  # [K, D]
            selected_sims = sims[indices_to_keep]

            weights = selected_sims
            weights = weights / weights.sum()

            aggregated_flat_grad = torch.sum(weights.unsqueeze(1) * selected_flat_grads, dim=0)

            selected_clf_grads = torch.stack(clf_grads)[indices_to_keep]
            weighted_clf_grad = torch.sum(weights.unsqueeze(1) * selected_clf_grads, dim=0)
            self.global_clf_grad_direction = self._normalize(weighted_clf_grad)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.round / self.total_rounds))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        print(f"[Aggregate Update] Round {self.round}, Global Learning Rate: {lr:.4f}")

        global_model_copy = copy.deepcopy(global_model)
        self._update_model_with_flat_grad(global_model_copy, aggregated_flat_grad, lr=lr)

        self.round += 1
        return global_model_copy.state_dict()
    def _update_model_with_flat_grad(self, model, flat_grad, lr=1.0):
        idx = 0
        with torch.no_grad():
            for param in model.parameters():
                num_params = param.numel()
                grad_part = flat_grad[idx:idx + num_params]
                grad_tensor = grad_part.view(param.shape)
                param.sub_(lr * grad_tensor)
                idx += num_params
                
                
class ShieldFL_GradientAggregator:
    def __init__(self, start_attack_epoch=10, device=None):
        self.global_clf_grad_history = None
        self.round = 0
        self.start_attack_epoch = start_attack_epoch
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _normalize(self, vec):
        norm = vec.norm()
        return vec / (norm + 1e-8)

    def _cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def aggregate(self, global_model, client_models=None, client_gradients=None, client_clf_gradients=None):
        if client_gradients is None:
            raise ValueError("Must provide 'client_gradients' for model update.")
        if client_clf_gradients is None:
            raise ValueError("ShieldFL requires 'client_clf_gradients' for defense.")

        raw_grads = [g.to(self.device) for g in client_gradients]        # (N, D)
        clf_grads = [g.to(self.device) for g in client_clf_gradients]    # (N, D_clf)
        n_clients = len(raw_grads)

        if n_clients == 0:
            return global_model.state_dict()

        norm_clf_grads = torch.stack([self._normalize(g) for g in clf_grads])  # (N, D_clf)

        global_model_copy = copy.deepcopy(global_model).to(self.device)

        if self.round < self.start_attack_epoch:
            avg_clf_grad = torch.mean(norm_clf_grads, dim=0)
            self.global_clf_grad_history = self._normalize(avg_clf_grad)
            print(f"[ShieldFL-SGD Round {self.round}] Pre-defense phase: using simple average for history")
            weights = torch.ones(n_clients, device=self.device) / n_clients

        else:
            sims_with_history = torch.stack([
                torch.dot(g, self.global_clf_grad_history) for g in norm_clf_grads
            ])
            sims_with_history_np = sims_with_history.cpu().numpy()
            print(f"[ShieldFL-SGD Round {self.round}] ClfGrad Similarity with history: {sims_with_history_np.round(4)}")

            outlier_idx = torch.argmin(sims_with_history).item()
            print(f"[ShieldFL-SGD] Client {outlier_idx} selected as outlier (sim={sims_with_history[outlier_idx]:.4f})")

            attack_base = norm_clf_grads[outlier_idx]
            sims_with_attack = torch.stack([
                torch.dot(g, attack_base) for g in norm_clf_grads
            ])
            sims_with_attack_np = sims_with_attack.cpu().numpy()
            print(f"[ShieldFL-SGD] Similarity with attack base: {sims_with_attack_np.round(4)}")

            weights = 1.0 - sims_with_attack
            weights = torch.clamp(weights, min=1e-8)
            weights = weights / weights.sum()
            print(f"[ShieldFL-SGD] Aggregation weights: {weights.cpu().numpy().round(4)}")

            weighted_avg_clf_grad = torch.sum(weights.unsqueeze(1) * norm_clf_grads, dim=0)
            self.global_clf_grad_history = self._normalize(weighted_avg_clf_grad)

        # raw_grads: list of (D,) tensors → stack to (N, D)
        raw_grads_stacked = torch.stack(raw_grads)  # (N, D)
        aggregated_grad = torch.sum(weights.unsqueeze(1) * raw_grads_stacked, dim=0)  # (D,)

        initial_lr = 0.5
        min_lr = 0.05
        total_rounds = 105
        current_round = self.round

        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_round / total_rounds))
        lr = 1
        print(f"[Aggregate Update] Round {current_round}, Global Learning Rate: {lr:.4f}")

        self._update_model_with_grad(global_model_copy, aggregated_grad, lr=lr)
        self.round += 1
        return {k: v.cpu() for k, v in global_model_copy.state_dict().items()}

    def _update_model_with_grad(self, model, flat_grad, lr=1.0):
        idx = 0
        with torch.no_grad():
            for param in model.parameters():
                num_params = param.numel()
                grad_part = flat_grad[idx:idx + num_params]
                grad_tensor = grad_part.reshape(param.shape)
                param.sub_(lr * grad_tensor)
                idx += num_params

    def _get_clf_slice(self, model, clf_layer_name='fc2.weight'):
        idx = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            if name == clf_layer_name:
                start = idx
                end = idx + num_params
                return slice(start, end)
            idx += num_params
        return None


class Krum_Aggregator:
    def __init__(self, f=1, multi_krum=True, start_decay_epoch=0, total_rounds=105,
                 initial_lr=0.5, min_lr=0.05):
        self.f = f
        self.multi_krum = multi_krum
        self.start_decay_epoch = start_decay_epoch
        self.total_rounds = total_rounds
        self.round = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_lr = initial_lr
        self.min_lr = min_lr

    def _cdist_sq(self, X):
        X_norm = (X ** 2).sum(1, keepdim=True)
        dist = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        return torch.clamp(dist, min=0.0)

    def _krum_scores_vectorized(self, flat_deltas):
        n = len(flat_deltas)

        matrix = torch.stack([d.to(self.device) for d in flat_deltas])
        dist_sq = self._cdist_sq(matrix)

        k = n - self.f - 2
        dist_sq.fill_diagonal_(float('inf'))

        nearest_k = torch.topk(dist_sq, k, dim=1, largest=False, sorted=False).values
        scores = nearest_k.sum(dim=1)
        return scores

    def _cosine_annealing_lr(self):
        if self.round < self.start_decay_epoch:
            return self.initial_lr
        decay_progress = (self.round - self.start_decay_epoch) / max(1, self.total_rounds - self.start_decay_epoch)
        decay_progress = max(0.0, min(1.0, decay_progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        return lr

    def aggregate(self, global_model, client_models, client_gradients=None):
        if not client_models:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model

        n = len(client_models)

        if hasattr(global_model, 'state_dict'):
            global_state = global_model.state_dict()
        else:
            global_state = global_model

        param_names = [k for k in global_state.keys() if global_state[k].ndim > 0]

        flat_deltas = []
        for state in client_models:
            delta_parts = []
            for k in param_names:
                delta = (state[k] - global_state[k].to(state[k].device)).view(-1)
                delta_parts.append(delta)
            flat_delta = torch.cat(delta_parts)
            flat_deltas.append(flat_delta)

        scores = self._krum_scores_vectorized(flat_deltas)
        _, sorted_indices = torch.topk(scores, k=n, largest=False)
        sorted_indices = sorted_indices.tolist()

        if self.multi_krum:
            selected_indices = sorted_indices[:n - self.f]
        else:
            selected_indices = [sorted_indices[0]]

        print(f"[KrumAggregator Round {self.round}] Krum scores: {scores.cpu().numpy().round(4)}")
        print(f"[KrumAggregator] Selected clients: {selected_indices}")

        lr = 0.5
        print(f"[KrumAggregator Round {self.round}] Learning Rate: {lr:.4f}")

        new_state = copy.deepcopy(global_state)
        for k in param_names:
            device = new_state[k].device
            delta_list = [client_models[i][k] - global_state[k].to(client_models[i][k].device) 
                         for i in selected_indices]
            avg_delta = torch.stack(delta_list).mean(dim=0)
            new_state[k] = global_state[k] + lr * avg_delta.to(device)

        self.round += 1
        return new_state


class TrimmedMean_Aggregator:
    def __init__(self, trim_ratio=0.1, start_decay_epoch=0, total_rounds=105,
                 initial_lr=0.5, min_lr=0.05):
        self.trim_ratio = trim_ratio
        self.start_decay_epoch = start_decay_epoch
        self.total_rounds = total_rounds
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.round = 0

    def _cosine_annealing_lr(self):
        if self.round < self.start_decay_epoch:
            return self.initial_lr
        decay_progress = (self.round - self.start_decay_epoch) / max(1, self.total_rounds - self.start_decay_epoch)
        decay_progress = max(0.0, min(1.0, decay_progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        return lr

    def aggregate(self, global_model, client_models, client_gradients=None):
        if not client_models:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        global_state = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        client_states = [
            cm.state_dict() if hasattr(cm, 'state_dict') else cm
            for cm in client_models
        ]

        device = list(global_state.values())[0].device
        param_names = [k for k in global_state.keys() if global_state[k].ndim > 0]
        shapes = {k: global_state[k].shape for k in param_names}

        deltas = []
        for state in client_states:
            delta = {}
            for k in param_names:
                g_param = global_state[k]
                c_param = state[k]
                delta[k] = (c_param.to(device) - g_param.to(device))
            deltas.append(delta)

        w_trimmed = self._trimmed_mean_gpu(deltas, param_names, shapes, device)

        lr = self._cosine_annealing_lr()
        print(f"[TrimmedMean Round {self.round}] Global Learning Rate: {lr:.4f}")

        new_state = copy.deepcopy(global_state)
        for k in param_names:
            update = lr * w_trimmed[k].to(new_state[k].device)
            new_state[k] = new_state[k] + update

        self.round += 1
        return new_state

    def _trimmed_mean_gpu(self, deltas, param_names, shapes, device):
        n_clients = len(deltas)
        if self.trim_ratio == 0.0:
            avg = {}
            for k in param_names:
                stacked = torch.stack([delta[k] for delta in deltas], dim=0)  # [N, ...]
                avg[k] = torch.mean(stacked, dim=0).to(device)
            return avg

        assert self.trim_ratio < 0.5, f"trim_ratio must be < 0.5, got {self.trim_ratio}"
        trim_num = int(self.trim_ratio * n_clients)

        w_result = {}
        for k in param_names:
            shape = shapes[k]
            total_num = reduce(lambda x, y: x * y, shape)
            flat_size = (n_clients, total_num)

            flat_deltas = torch.stack([
                delta[k].reshape(-1) for delta in deltas
            ]).to(device)

            y = flat_deltas.t()

            y_sorted = torch.sort(y, dim=1)[0]

            if trim_num > 0:
                y_trimmed = y_sorted[:, trim_num:-trim_num]
            else:
                y_trimmed = y_sorted

            result_flat = torch.mean(y_trimmed, dim=1)

            w_result[k] = result_flat.reshape(shape)

        return w_result


class FedAvg_Aggregator:
    def __init__(self):
        self.round = 0

    def aggregate(self, global_model, client_models, client_gradients=None):
        if not client_models:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model

        if hasattr(global_model, 'state_dict'):
            global_state = global_model.state_dict()
        else:
            global_state = global_model

        client_states = []
        for cm in client_models:
            if hasattr(cm, 'state_dict'):
                client_states.append(cm.state_dict())
            else:
                client_states.append(cm)

        device = list(global_state.values())[0].device
        param_names = list(global_state.keys())

        new_state = copy.deepcopy(global_state)
        for k in param_names:
            if global_state[k].ndim == 0:
                continue
            client_params = [state[k].to(device) for state in client_states]
            avg_param = sum(client_params) / len(client_params)
            new_state[k] = avg_param.to(global_state[k].device)

        self.round += 1
        return new_state
    
    
    
class FLTrust_Aggregator:
    def __init__(self, server_clean_loader, model_fn, device,
                 clf_layer_name='fc2.weight',
                 use_server_anchor=True):
        self.server_clean_loader = server_clean_loader
        self.model_fn = model_fn
        self.device = device
        self.clf_layer_name = clf_layer_name
        self.use_server_anchor = use_server_anchor
        self.round = 0

    def _extract_clf_grad(self, model, loss):
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if name == self.clf_layer_name and param.grad is not None:
                grad = param.grad.data.view(-1)
                return (-grad) / (grad.norm() + 1e-8)
        raise ValueError(f"Classifier layer '{self.clf_layer_name}' not found or has no gradient.")

    def compute_trust_direction(self, global_model, num_batches=5):
        model = self.model_fn().to(self.device)
        model.load_state_dict(copy.deepcopy(global_model))
        model.train()

        total_grad = None
        count = 0
        loader_iter = iter(self.server_clean_loader)

        for _ in range(num_batches):
            try:
                data, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.server_clean_loader)
                data, targets = next(loader_iter)

            data, targets = data.to(self.device), targets.to(device=self.device)
            output = model(data)
            loss = F.cross_entropy(output, targets)

            try:
                clf_grad_dir = self._extract_clf_grad(model, loss)
                if total_grad is None:
                    total_grad = clf_grad_dir.clone()
                else:
                    total_grad += clf_grad_dir
                count += 1
            except ValueError as e:
                print(f"[WARNING] {e}")
                continue

        if count == 0:
            raise RuntimeError("No valid batch processed in compute_trust_direction")

        avg_dir = total_grad / count
        return avg_dir / (avg_dir.norm() + 1e-8)

    def aggregate(self, global_model, client_models):
        device = self.device
        trust_dir = self.compute_trust_direction(global_model).to(device)  # (D_clf,)
        param_names = [k for k in global_model.keys() if 'num_batches_tracked' not in k]

        clf_layer_name = self.clf_layer_name

        similarities = []
        for client_state in client_models:
            if clf_layer_name not in client_state:
                raise ValueError(f"Client model missing {clf_layer_name}")

            global_param = global_model[clf_layer_name].to(device)
            client_param = client_state[clf_layer_name].to(device)
            delta = (client_param - global_param).view(-1)
            norm = delta.norm()
            if norm < 1e-8:
                print("[Warning] Client update norm is zero.")
                unit_update = torch.zeros_like(delta)
            else:
                unit_update = delta / norm

            # 计算余弦相似度
            cos_sim = F.cosine_similarity(
                trust_dir.unsqueeze(0),
                unit_update.unsqueeze(0),
                dim=1
            ).item()

            similarities.append(max(0.0, cos_sim))

        total_weight = sum(similarities)
        if total_weight < 1e-8:
            print("[FLTrust] All similarities are zero. Using uniform averaging.")
            weights = [1.0 / len(similarities)] * len(similarities)
        else:
            weights = [w / total_weight for w in similarities]

        new_state = {}
        for k in param_names:
            weighted_sum = torch.zeros_like(global_model[k], device=device)
            for i, client_state in enumerate(client_models):
                weight = weights[i]
                weighted_sum += weight * client_state[k].to(device)
            new_state[k] = weighted_sum

        if self.use_server_anchor:
            alpha0 = 1.0
            for k in param_names:
                if k in new_state:
                    new_state[k] = (alpha0 * global_model[k].to(device) + new_state[k]) / (alpha0 + 1.0)

        self.round += 1
        return new_state

class FoolsGoldmod:
    def __init__(self, num_peers, clf_layer_name="fc2.weight", model_fn=None, device=None, use_server_anchor=False):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
        self.clf_layer_name = clf_layer_name
        self.clf_slice = None
        self.model_fn = model_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        self.use_server_anchor = use_server_anchor

    def _get_clf_slice(self, model, clf_layer_name):
        idx = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            if name == clf_layer_name:
                return slice(idx, idx + num_params)
            idx += num_params
        raise ValueError(f"Classifier layer '{clf_layer_name}' not found in model.")

    def aggregate(self, global_model, client_models, selected_peers):
        if len(client_models) == 0:
            raise ValueError("No client models provided.")
        device = self.device
        m = len(client_models)

        if self.clf_slice is None:
            if self.model_fn is None:
                raise ValueError("model_fn must be provided in the first round to determine clf_slice.")
            temp_model = self.model_fn().to(device)
            self.clf_slice = self._get_clf_slice(temp_model, self.clf_layer_name)
            print(f"[FoolsGold] Using classifier layer: {self.clf_layer_name}, slice: {self.clf_slice}")

        clf_deltas = []
        for client_state in client_models:
            if self.clf_layer_name not in client_state:
                raise ValueError(f"Client model missing {self.clf_layer_name}")
            global_param = global_model[self.clf_layer_name].to(device)
            client_param = client_state[self.clf_layer_name].to(device)
            delta = (client_param - global_param).view(-1)
            clf_deltas.append(delta)

        clf_deltas_tensor = torch.stack(clf_deltas)
        grad_len = clf_deltas_tensor.shape[1]


        if self.memory is None:
            self.memory = torch.zeros((self.num_peers, grad_len), device=device)
        elif self.memory.device != device:
            self.memory = self.memory.to(device)

        selected_peers_tensor = torch.LongTensor(selected_peers).to(device)
        self.memory.index_add_(0, selected_peers_tensor, clf_deltas_tensor)
        memory_selected = self.memory[selected_peers_tensor]

        wv = self._foolsgold_torch(memory_selected)
        self.wv_history.append(wv.detach().cpu().numpy().copy())

        wv = wv / wv.sum()

        param_names = [k for k in global_model.keys() if 'num_batches_tracked' not in k]
        new_state = {}
        for k in param_names:
            weighted_sum = torch.zeros_like(global_model[k], device=device)
            for i, client_state in enumerate(client_models):
                weighted_sum += wv[i] * client_state[k].to(device)
            new_state[k] = weighted_sum

        if self.use_server_anchor:
            alpha0 = 1.0
            for k in param_names:
                new_state[k] = (alpha0 * global_model[k].to(device) + new_state[k]) / (alpha0 + 1.0)

        return new_state

    def _foolsgold_torch(self, grads):
        n = grads.size(0)
        if n == 1:
            return torch.ones(1, device=grads.device)
        with torch.no_grad():
            norms = grads.norm(p=2, dim=1, keepdim=True)
            grads_normalized = grads / (norms + 1e-8)
            sim_matrix = torch.mm(grads_normalized, grads_normalized.t())
            sim_matrix = torch.clamp(sim_matrix, -1, 1)
            fg_matrix = 1 - sim_matrix
            v = torch.max(fg_matrix, dim=1)[0]
            v = torch.clamp(v, min=1e-8)
            wv = 1.0 / v
            wv = wv / wv.sum() * n
        return wv
    
    


class RPPD:
    def __init__(self, start_defense_epoch=5, attackers_ratio=0.1,
                 initial_lr=0.1, min_lr=0.01, total_rounds=105, clf_layer_name="fc2.weight"):
        self.global_clf_delta_direction = None
        self.round = 0
        self.attackers_ratio = attackers_ratio
        self.start_defense_epoch = start_defense_epoch
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_rounds = total_rounds
        self.clf_layer_name = clf_layer_name
        self.clf_slice = None

    def _normalize(self, vec):
        norm = vec.norm(p=2)
        return vec / (norm + 1e-8)

    def _get_clf_slice(self, model_state, clf_layer_name):
        idx = 0
        for name in model_state.keys():
            if not isinstance(model_state[name], torch.Tensor) or model_state[name].numel() == 0:
                continue
            num_params = model_state[name].numel()
            if name == clf_layer_name:
                start = idx
                end = idx + num_params
                return slice(start, end)
            idx += num_params
        raise ValueError(f"Classifier layer '{clf_layer_name}' not found in model state_dict.")

    def aggregate(self, global_model, client_models=None, client_gradients=None):
        if client_models is None or len(client_models) == 0:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model

        if hasattr(global_model, 'state_dict'):
            global_state = global_model.state_dict()
        else:
            global_state = global_model

        device = next(iter(global_state.values())).device if hasattr(next(iter(global_state.values())), 'device') else torch.device('cpu')
        n_clients = len(client_models)
        alpha = int(self.attackers_ratio * n_clients)
        if alpha >= n_clients:
            raise ValueError(f"Too many attackers: alpha={alpha}, n_clients={n_clients}")

        if self.clf_slice is None:
            self.clf_slice = self._get_clf_slice(global_state, self.clf_layer_name)
            print(f"[ClassifierOnly] Using classifier layer: {self.clf_layer_name}, slice: {self.clf_slice}")
        flat_deltas = []
        for client_state in client_models:
            delta_parts = []
            for k in global_state.keys():
                if not isinstance(global_state[k], torch.Tensor) or global_state[k].numel() == 0:
                    continue
                client_param = client_state[k].to(device)
                global_param = global_state[k].to(device)
                delta = (client_param - global_param).view(-1)
                delta_parts.append(delta)
            flat_delta = torch.cat(delta_parts)
            flat_deltas.append(flat_delta)

        clf_deltas = [d[self.clf_slice] for d in flat_deltas]
        normalized_clf_deltas = [self._normalize(d) for d in clf_deltas]

        if self.round == 0 or self.round < self.start_defense_epoch or self.global_clf_delta_direction is None:
            print(f"[Round {self.round}] Warm-up: simple average to initialize direction.")
            avg_clf_delta = torch.stack(clf_deltas).mean(dim=0)
            self.global_clf_delta_direction = self._normalize(avg_clf_delta)
            selected_client_states = client_models
        else:
            current_clf_directions = torch.stack(normalized_clf_deltas)
            sims = torch.mv(current_clf_directions, self.global_clf_delta_direction)
            sims_np = sims.cpu().numpy()
            print(f"[Round {self.round}] ClfDelta Cosine similarities: {sims_np.round(4)}")

            _, indices_to_keep = torch.topk(sims, n_clients - alpha, largest=True)
            exclude_indices = [i for i in range(n_clients) if i not in indices_to_keep]
            print(f"[RPPD] Excluding clients: {exclude_indices}")

            selected_client_states = [client_models[i] for i in indices_to_keep]

            selected_clf_deltas = [clf_deltas[i] for i in indices_to_keep]
            selected_sims = sims[indices_to_keep]
            weights = selected_sims / selected_sims.sum()
            weighted_clf_delta = torch.sum(torch.stack(selected_clf_deltas) * weights.unsqueeze(1), dim=0)
            self.global_clf_delta_direction = self._normalize(weighted_clf_delta)

        flat_models = []
        for client_state in selected_client_states:
            model_parts = []
            for k in global_state.keys():
                if not isinstance(global_state[k], torch.Tensor) or global_state[k].numel() == 0:
                    continue
                client_param = client_state[k].to(device)
                model_parts.append(client_param.view(-1))
            flat_model = torch.cat(model_parts)
            flat_models.append(flat_model)

        aggregated_flat_model = torch.stack(flat_models).mean(dim=0)

        new_state = copy.deepcopy(global_state)
        idx = 0
        with torch.no_grad():
            for k in new_state.keys():
                if not isinstance(new_state[k], torch.Tensor) or new_state[k].numel() == 0:
                    continue
                num_params = new_state[k].numel()
                model_part = aggregated_flat_model[idx:idx + num_params]
                model_tensor = model_part.view(new_state[k].shape).to(new_state[k].device)
                new_state[k] = model_tensor
                idx += num_params

        self.round += 1
        return new_state
    
class ShieldFL_mod: 
    def __init__(self, start_attack_epoch=1, device=None, clf_layer_name='fc2.weight'):
        self.global_clf_grad_history = None
        self.round = 0
        self.start_attack_epoch = 1
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf_layer_name = clf_layer_name
        self.clf_slice = None

    def _normalize(self, vec):
        norm = vec.norm()
        return vec / (norm + 1e-8)

    def _cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def _get_clf_slice(self, model_state):
        idx = 0
        for name in model_state.keys():
            if not isinstance(model_state[name], torch.Tensor) or model_state[name].numel() == 0:
                continue
            num_params = model_state[name].numel()
            if name == self.clf_layer_name:
                start = idx
                end = idx + num_params
                return slice(start, end)
            idx += num_params
        raise ValueError(f"Classifier layer '{self.clf_layer_name}' not found.")

    def aggregate(self, global_model, client_models=None, client_gradients=None, client_clf_gradients=None):
        if client_models is None or len(client_models) == 0:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model

        # 获取全局模型状态
        if hasattr(global_model, 'state_dict'):
            global_state = global_model.state_dict()
        else:
            global_state = global_model

        device = next(iter(global_state.values())).device if hasattr(next(iter(global_state.values())), 'device') else self.device
        n_clients = len(client_models)

        if self.clf_slice is None:
            self.clf_slice = self._get_clf_slice(global_state)
            print(f"[ShieldFL-Diff] Classifier layer '{self.clf_layer_name}' slice: {self.clf_slice}")
        client_deltas = []
        clf_deltas = []

        for client_state in client_models:
            delta_parts = []
            for k in global_state.keys():
                if not isinstance(global_state[k], torch.Tensor) or global_state[k].numel() == 0:
                    continue
                client_param = client_state[k].to(device)
                global_param = global_state[k].to(device)
                delta = (client_param - global_param).view(-1)
                delta_parts.append(delta)
            flat_delta = torch.cat(delta_parts)
            client_deltas.append(flat_delta)
            clf_deltas.append(flat_delta[self.clf_slice])

        norm_clf_grads = torch.stack([self._normalize(g) for g in clf_deltas])

        global_model_copy = copy.deepcopy(global_model)
        if hasattr(global_model_copy, 'to'):
            global_model_copy = global_model_copy.to(device)

        if self.round < self.start_attack_epoch:
            avg_clf_grad = torch.mean(norm_clf_grads, dim=0)
            self.global_clf_grad_history = self._normalize(avg_clf_grad)
            print(f"[ShieldFL-Diff Round {self.round}] Pre-defense phase: using simple average for history")
            weights = torch.ones(n_clients, device=device) / n_clients

        else:
            sims_with_history = torch.stack([
                torch.dot(g, self.global_clf_grad_history) for g in norm_clf_grads
            ])
            sims_with_history_np = sims_with_history.cpu().numpy()
            print(f"[ShieldFL-Diff Round {self.round}] ClfGrad Similarity with history: {sims_with_history_np.round(4)}")
            outlier_idx = torch.argmin(sims_with_history).item()
            print(f"[ShieldFL-Diff] Client {outlier_idx} selected as outlier (sim={sims_with_history[outlier_idx]:.4f})")

            attack_base = norm_clf_grads[outlier_idx]
            sims_with_attack = torch.stack([
                torch.dot(g, attack_base) for g in norm_clf_grads
            ])
            sims_with_attack_np = sims_with_attack.cpu().numpy()
            print(f"[ShieldFL-Diff] Similarity with attack base: {sims_with_attack_np.round(4)}")

            weights = 1.0 - sims_with_attack
            weights = torch.clamp(weights, min=1e-8)
            weights = weights / weights.sum()
            print(f"[ShieldFL-Diff] Aggregation weights: {weights.cpu().numpy().round(4)}")

            weighted_avg_clf_grad = torch.sum(weights.unsqueeze(1) * norm_clf_grads, dim=0)
            self.global_clf_grad_history = self._normalize(weighted_avg_clf_grad)

        raw_grads_stacked = torch.stack(client_deltas)
        aggregated_grad = torch.sum(weights.unsqueeze(1) * raw_grads_stacked, dim=0)
        initial_lr = 0.5
        min_lr = 0.05
        total_rounds = 105
        current_round = self.round

        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_round / total_rounds))
        lr = 0.5
        print(f"[Aggregate Update] Round {current_round}, Global Learning Rate: {lr:.4f}")
        self._update_model_with_delta(global_model_copy, aggregated_grad, lr=lr)
        self.round += 1

        if hasattr(global_model_copy, 'state_dict'):
            return {k: v.cpu() for k, v in global_model_copy.state_dict().items()}
        else:
            return {k: v.cpu() for k, v in global_model_copy.items()}

    def _update_model_with_delta(self, model, flat_delta, lr=1.0):
        idx = 0
        with torch.no_grad():
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    num_params = param.numel()
                    delta_part = flat_delta[idx:idx + num_params]
                    delta_tensor = delta_part.reshape(param.shape).to(param.device)
                    param.add_(lr * delta_tensor) 
                    idx += num_params
            else:
                for k in model.keys():
                    if not isinstance(model[k], torch.Tensor) or model[k].numel() == 0:
                        continue
                    num_params = model[k].numel()
                    delta_part = flat_delta[idx:idx + num_params]
                    delta_tensor = delta_part.reshape(model[k].shape).to(model[k].device)
                    model[k] = model[k] + lr * delta_tensor 
                    idx += num_params