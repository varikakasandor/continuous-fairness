import torch
import numpy as np
from torch import nn


def generate_per_category_losses(constrained_intervals_A, quantizition_intervals_Y, return_category_names=False):
    def per_category_losses(Y_hat, A, Y):
        losses = []
        categories = []
        for Y_start, Y_end in quantizition_intervals_Y:
            for A_start, A_end in constrained_intervals_A:
                y_a_mask = (((Y_start < Y) & (Y <= Y_end)) & (A_start < A) & (A <= A_end))  # always < first <= later!
                if y_a_mask.any():
                    masked_prediction = Y_hat[y_a_mask]
                    masked_target = Y[y_a_mask]
                    curr_obj_loss = nn.BCELoss()(masked_prediction, masked_target)
                    losses.append(curr_obj_loss)
                    if return_category_names:
                        categories.append(((Y_start, Y_end), (A_start, A_end)))
        if return_category_names:
            return losses, categories
        else:
            return torch.stack(losses)

    return per_category_losses


def generate_alpha(constrained_intervals_A, quantizition_intervals_Y, return_category_names=False):
    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        categories = []
        for Y_start, Y_end in quantizition_intervals_Y:
            y_mask = ((Y_start < Y) & (Y <= Y_end))
            cnt_y = y_mask.sum()
            for A_start, A_end in constrained_intervals_A:
                y_a_mask = (y_mask & (A_start < A) & (A <= A_end))  # always < first <= later!
                cnt_y_a = y_a_mask.sum()
                sum_y_yhat = (y_mask * Y_hat).sum()
                sum_y_a_yhat = (y_a_mask * Y_hat).sum()
                if cnt_y_a > 0 and cnt_y > 0:
                    curr_nd_loss = torch.abs(sum_y_a_yhat / cnt_y_a - sum_y_yhat / cnt_y)
                    nd_losses.append(curr_nd_loss)
                    if return_category_names:
                        categories.append(((Y_start, Y_end), (A_start, A_end)))
        if return_category_names:
            return nd_losses, categories
        else:
            return torch.stack(nd_losses)

    return fairness_metric


def generate_beta(constrained_intervals_A, quantizition_intervals_Y, size_compensation=lambda x: np.sqrt(x)):
    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        n = len(Y_hat)
        for Y_start, Y_end in quantizition_intervals_Y:
            y_mask = ((Y_start < Y) & (Y <= Y_end))
            cnt_y = y_mask.sum()
            for A_start, A_end in constrained_intervals_A:
                y_a_mask = (y_mask & (A_start < A) & (A <= A_end))  # always < first <= later!
                cnt_y_a = y_a_mask.sum()
                sum_y_yhat = (y_mask * Y_hat).sum()
                sum_y_a_yhat = (y_a_mask * Y_hat).sum()
                if cnt_y_a > 0 and cnt_y > 0:
                    curr_nd_loss = torch.abs(sum_y_a_yhat / cnt_y_a - sum_y_yhat / cnt_y) * size_compensation(
                        cnt_y_a / n)
                    nd_losses.append(curr_nd_loss)
        return torch.stack(nd_losses)

    return fairness_metric


def generate_constrained_intervals(num_constrained_intervals):
    endpoints = np.linspace(-0.0001, 1.0001, num_constrained_intervals + 1)
    constrained_intervals = []
    for i in range(len(endpoints) - 1):
        constrained_intervals.append((endpoints[i], endpoints[i + 1]))
    return constrained_intervals
