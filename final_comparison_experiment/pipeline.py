import os
import sys

sys.path.append(os.path.abspath(os.path.join('../..')))
import torch
from torch import nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt

from models import SimpleNN


class FairnessAwareLearningExperiment:
    def __init__(self, data, fairness_metric, fairness_name, dataset_name, fairness_weights, analysis_metric, print_progress=True):
        self.x_train, self.y_train, self.a_train, self.x_test, self.y_test, self.a_test = data
        self.fairness_metric = fairness_metric
        self.fairness_name = fairness_name
        self.dataset_name = dataset_name
        self.fairness_weights = fairness_weights
        self.print_progress = print_progress
        self.analysis_metric = analysis_metric

    def train_model(self, model, fairness_weight=1.0, lr=1e-5, num_epochs=5):
        X = torch.tensor(self.x_train.astype(np.float32))
        A = torch.tensor(self.a_train.astype(np.float32))
        Y = torch.tensor(self.y_train.astype(np.float32))
        dataset = data_utils.TensorDataset(X, Y, A)
        dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

        data_fitting_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        for j in range(num_epochs):
            if self.print_progress:
                print(f"EPOCH {j} started")
            for i, (x, y, a) in enumerate(dataset_loader):
                def closure():
                    optimizer.zero_grad()
                    prediction = model(x).flatten()
                    loss = fairness_weight * torch.mean(self.fairness_metric(prediction, a, y)) + data_fitting_loss(
                        prediction, y)
                    loss.backward()
                    return loss

                optimizer.step(closure)
            if self.print_progress:
                mse_curr_test, nd_curr_test = self.evaluate(model, dataset="test")
                mse_curr_train, nd_curr_train = self.evaluate(model, dataset="train")
                print(
                    f"TEST -- mse: {mse_curr_test}, nd: {nd_curr_test}, combined: {mse_curr_test + fairness_weight * nd_curr_test}")
                print(
                    f"TRAIN -- mse: {mse_curr_train}, nd: {nd_curr_train}, combined: {mse_curr_train + fairness_weight * nd_curr_train}")

    def evaluate(self, model, dataset="test"):
        if dataset == "train":
            X, A, Y = torch.tensor(self.x_train.astype(np.float32)), torch.Tensor(
                self.a_train.astype(np.float32)), torch.tensor(self.y_train.astype(np.float32))
        elif dataset == "test":
            X, A, Y = torch.tensor(self.x_test.astype(np.float32)), torch.Tensor(
                self.a_test.astype(np.float32)), torch.tensor(self.y_test.astype(np.float32))
        prediction = model(X).detach().flatten()
        loss = nn.MSELoss()(prediction, Y)
        nd_loss = torch.max(self.fairness_metric(prediction, A, Y))
        return loss.item(), nd_loss

    def run_analysis(self):
        X, A, Y = torch.tensor(self.x_test.astype(np.float32)), torch.Tensor(
            self.a_test.astype(np.float32)), torch.tensor(self.y_test.astype(np.float32))
        objective_losses = []
        nd_losses = []
        categories = None
        for fairness_weight in self.fairness_weights:
            if self.print_progress:
                print(f"Fairness weight {fairness_weight} started")
            model = SimpleNN(self.x_train.shape[1], 1)
            self.train_model(model, fairness_weight=fairness_weight)
            prediction = model(X).detach().flatten()
            loss = nn.MSELoss()(prediction, Y)
            curr_fairness_losses, category_names = self.analysis_metric(prediction, A, Y)
            if categories is None:
                categories = category_names
            objective_losses.append(loss)
            nd_losses.append(curr_fairness_losses)
        objective_losses, nd_losses = np.array(objective_losses), np.array(nd_losses)
        num_categories = nd_losses.shape[1]
        fig, axes = plt.subplots(1, num_categories, figsize=(16, 4))
        for i in range(num_categories):
            axes[i].scatter(nd_losses[:, i], objective_losses)
            axes[i].set_xlabel('Discrimimnatory loss')
            axes[i].set_ylabel('Objective loss')
            axes[i].set_title(categories[i])
        fig.suptitle(self.fairness_name)
        plt.tight_layout()
        plt.savefig(f'analysis_{self.fairness_name}_{self.dataset_name}.pdf')
        plt.show()
