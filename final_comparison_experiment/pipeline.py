import os
import sys
from enum import Enum
from datetime import datetime
import json

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torch import nn

from models import SimpleNN

sys.path.append(os.path.abspath(os.path.join('./..')))
from final_comparison_experiment.tools import *


class MySoftmax:
    def __init__(self, temp):
        self._temp = temp
        self._weight_grads = False

    def __call__(self, tensor):
        tensor_avg = tensor * tensor.sum()
        weights = torch.nn.functional.softmax(tensor_avg * self._temp, dim=0)
        if not self._weight_grads:
            weights = weights.detach()
        weighted_max = weights @ tensor
        return weighted_max


class MaxLosses(Enum):
    MAX = torch.max
    MEAN = torch.mean
    SOFTMAX = MySoftmax(10)


CUSTOM_MAX = MaxLosses.MAX
CUSTOM_MAX_NAME = CUSTOM_MAX.name
CUSTOM_MAX_FUN = CUSTOM_MAX.value


class FairnessAwareLearningExperiment:
    def __init__(self, data, fairness_metric, fairness_name, dataset_name, fairness_weights, analysis_metric, per_category_loss_function, lr,
                 num_epochs=100, print_progress=False, external_params={}):
        x_train, y_train, a_train, x_test, y_test, a_test = data
        self.x_train, self.y_train, self.a_train, self.x_test, self.y_test, self.a_test = torch.tensor(
            x_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)), torch.tensor(
            a_train.astype(np.float32)), torch.tensor(x_test.astype(np.float32)), torch.tensor(
            y_test.astype(np.float32)), torch.tensor(a_test.astype(np.float32))
        self.fairness_metric = fairness_metric
        self.fairness_name = fairness_name
        self.dataset_name = dataset_name
        self.fairness_weights = fairness_weights
        self.print_progress = print_progress
        self.analysis_metric = analysis_metric
        self.per_category_loss_function = per_category_loss_function
        self.num_epochs = num_epochs
        self.lr = lr
        self.external_params = external_params

    @property
    def _params(self):
        params = {
            'fairness_name': self.fairness_name.split('_')[0],
            'dataset_name': self.dataset_name,
            'num_epochs': self.num_epochs,
            'learning_rate': self.lr,
            'CUSTOM_MAX': CUSTOM_MAX_NAME
        }
        params.update(self.external_params)
        return params

    def train_model(self, model, fairness_weight=1.0):
        num_epochs = self.num_epochs
        dataset = data_utils.TensorDataset(self.x_train, self.y_train, self.a_train)
        dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

        data_fitting_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.01)

        for j in range(num_epochs):
            if self.print_progress:
                print(f"EPOCH {j + 1} started")
            for i, (x, y, a) in enumerate(dataset_loader):
                def closure():
                    optimizer.zero_grad()
                    prediction = model(x).flatten()
                    loss = fairness_weight * CUSTOM_MAX_FUN(self.fairness_metric(prediction, a, y)) + data_fitting_loss(
                        prediction, y)
                    loss.backward()
                    return loss

                optimizer.step(closure)
            if self.print_progress:
                bce_curr_train, alpha_curr_train, nd_curr_train = self.evaluate(model, dataset="train")
                bce_curr_test, alpha_curr_test, nd_curr_test = self.evaluate(model, dataset="test")
                print(
                    f"TEST -- L: {bce_curr_test}, alpha: {alpha_curr_test}, nd: {nd_curr_test}, combined: {bce_curr_test + fairness_weight * nd_curr_test}")
                print(
                    f"TRAIN -- loss: {bce_curr_train}, alpha: {alpha_curr_train}, nd: {nd_curr_train}, combined: {bce_curr_train + fairness_weight * nd_curr_train}")

    def evaluate(self, model, dataset="test"):
        x, a, y = (self.x_train, self.a_train, self.y_train) if dataset == "train" else (
            self.x_test, self.a_test, self.y_test)
        prediction = model(x).detach().flatten()
        loss = nn.BCELoss()(prediction, y)
        fairness_losses = self.fairness_metric(prediction, a, y)
        tmp, _ = self.analysis_metric(prediction, a, y)
        alpha_loss = torch.max(tmp)
        nd_loss = CUSTOM_MAX_FUN(fairness_losses)
        return loss.item(), alpha_loss, nd_loss

    def run_analysis(self):
        objective_losses_train, objective_losses_test = [], []
        per_category_losses_train, per_category_losses_test = [], []
        alpha_losses_train, alpha_losses_test = [], []
        categories = None
        bottlenecks_train, bottlenecks_test = [], []
        for fairness_weight in self.fairness_weights:
            print(f"Fairness weight {fairness_weight} started")
            model = SimpleNN(self.x_train.shape[1], 1)
            self.train_model(model, fairness_weight=fairness_weight)

            prediction_train, prediction_test = model(self.x_train).detach().flatten(), model(
                self.x_test).detach().flatten()
            objective_loss_train, objective_loss_test = nn.BCELoss()(prediction_train, self.y_train), nn.BCELoss()(
                prediction_test,
                self.y_test)
            per_category_loss_train, per_category_loss_test = self.per_category_loss_function(prediction_train,
                                                                                             self.a_train,
                                                                                             self.y_train), self.per_category_loss_function(
                prediction_test, self.a_test, self.y_test)
            alpha_loss_train, categories_train = self.analysis_metric(prediction_train, self.a_train,
                                                                      self.y_train)  # categories is the same for all iterations, should be restructured
            alpha_loss_test, categories_test = self.analysis_metric(prediction_test, self.a_test, self.y_test)
            assert categories_train == categories_test

            categories = categories_train
            curr_bottleneck_train, curr_bottleneck_test = [1] * len(alpha_loss_train), [1] * len(alpha_loss_test)
            curr_bottleneck_train[alpha_loss_train.index(max(alpha_loss_train))], curr_bottleneck_test[
                alpha_loss_test.index(max(alpha_loss_test))] = MARKER_STRONG_WIDTH, MARKER_STRONG_WIDTH
            bottlenecks_train.append(curr_bottleneck_train)
            bottlenecks_test.append(curr_bottleneck_test)
            objective_losses_train.append(objective_loss_train)
            objective_losses_test.append(objective_loss_test)
            per_category_losses_train.append(per_category_loss_train)
            per_category_losses_test.append(per_category_loss_test)
            alpha_losses_train.append(alpha_loss_train)
            alpha_losses_test.append(alpha_loss_test)

            # -- Saves run meta data
            run_records = self._params
            local_params = {
                'fairness_weight': fairness_weight
            }
            run_records.update(local_params)
            results_dict = {
                'objective_loss_train': float(objective_loss_train),
                'loss_train_00': float(per_category_loss_train[0]),
                'loss_train_01': float(per_category_loss_train[1]),
                'loss_train_10': float(per_category_loss_train[2]),
                'loss_train_11': float(per_category_loss_train[3]),
                'objective_loss_test': float(objective_loss_test),
                'loss_test_00': float(per_category_loss_test[0]),
                'loss_test_01': float(per_category_loss_test[1]),
                'loss_test_10': float(per_category_loss_test[2]),
                'loss_test_11': float(per_category_loss_test[3])
            }
            """results_dict = {
                'objective_loss_train': float(objective_loss_train),
                'alpha_loss_train_00': float(alpha_loss_train[0]),
                'alpha_loss_train_01': float(alpha_loss_train[1]),
                'alpha_loss_train_10': float(alpha_loss_train[2]),
                'alpha_loss_train_11': float(alpha_loss_train[3]),
                'objective_loss_test': float(objective_loss_test),
                'alpha_loss_test_00': float(alpha_loss_test[0]),
                'alpha_loss_test_01': float(alpha_loss_test[1]),
                'alpha_loss_test_10': float(alpha_loss_test[2]),
                'alpha_loss_test_11': float(alpha_loss_test[3])
            }"""
            run_records.update(results_dict)
            filename = f'records/run_{datetime.now().timestamp()}.json'
            with open(filename, 'w') as file:
                json.dump(run_records, file)

        objective_losses_train, objective_losses_test, alpha_losses_train, alpha_losses_test, per_category_losses_train, per_category_losses_test = np.array(
            objective_losses_train), np.array(objective_losses_test), np.array(alpha_losses_train), np.array(
            alpha_losses_test), np.array(per_category_losses_train), np.array(per_category_losses_test)

        num_categories = len(categories)
        plot_dim_1, plot_dim_2 = find_optimal_subplot_dims(num_categories)
        fig, axes = plt.subplots(plot_dim_1, plot_dim_2, figsize=(plot_dim_2 * 4, plot_dim_1 * 4))

        for i in range(num_categories):
            idx, idy = i // plot_dim_2, i % plot_dim_2
            scatter_train = axes[idx, idy].scatter(per_category_losses_train[:, i], objective_losses_train,
                                                   c='green', linewidths=[l[i] for l in bottlenecks_train],
                                                   label='train')
            scatter_test = axes[idx, idy].scatter(per_category_losses_test[:, i], objective_losses_test,
                                                  c='blue', linewidths=[l[i] for l in bottlenecks_test], label='test')
            axes[idx, idy].legend(handles=[scatter_train, scatter_test])
            axes[idx, idy].set_xlabel('Per category objective loss')
            axes[idx, idy].set_ylabel('Overall objective loss')
            (Y_start, Y_end), (A_start, A_end) = categories[i]
            category_prob_train = ((Y_start < self.y_train) & (self.y_train <= Y_end) & (A_start < self.a_train) & (
                    self.a_train <= A_end)).sum() / len(self.y_train)
            category_prob_test = ((Y_start < self.y_test) & (self.y_test <= Y_end) & (A_start < self.a_test) & (
                    self.a_test <= A_end)).sum() / len(self.y_test)
            category_desc = f"Y: ({Y_start:.3f} - {Y_end:.3f}), A: ({A_start:.3f} - {A_end:.3f}), P_ya_train: {category_prob_train:.3f}, P_ya_test: {category_prob_test:.3f}"
            axes[idx, idy].set_title(category_desc, fontsize="xx-small")

        fig.suptitle(self.fairness_name)
        plt.tight_layout()
        plt.savefig(f'./plots/analysis_{self.fairness_name}.pdf')

        return ExperimentResults(self.y_train, self.a_train, self.y_test, self.a_test, categories,
                                 objective_losses_train, alpha_losses_train, per_category_losses_train, bottlenecks_train, objective_losses_test,
                                 alpha_losses_test, per_category_losses_test, bottlenecks_test, self.fairness_name, self.dataset_name)
