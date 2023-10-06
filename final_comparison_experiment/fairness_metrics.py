import torch
import numpy as np


def generate_alpha(constrained_intervals_A, quantizition_intervals_Y, return_category_names = False):
    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        for Y_start, Y_end in quantizition_intervals_Y:
            for A_start, A_end in constrained_intervals_A:
                y_a_mask = ((Y_start <= Y) & (Y < Y_end) & (A_start <= A) & (A < A_end))
                y_mask = ((Y_start <= Y) & (Y < Y_end))
                cnt_y_a = y_a_mask.sum()
                cnt_y = y_mask.sum()
                sum_y_yhat = (y_mask * Y_hat).sum()
                sum_y_a_yhat = (y_a_mask * Y_hat).sum()
                if cnt_y_a > 0 and cnt_y > 0:
                    curr_nd_loss = torch.abs(sum_y_a_yhat / cnt_y_a - sum_y_yhat / cnt_y)
                    nd_losses.append(curr_nd_loss)
        if return_category_names:
            categories = []
            for inter_Y in quantizition_intervals_Y:
                for inter_A in constrained_intervals_A:
                    categories.append((inter_Y, inter_A))
            return nd_losses, categories
        else:
            return torch.stack(nd_losses)

    return fairness_metric


def generate_beta(constrained_intervals_A, quantizition_intervals_Y, size_compensation=lambda x: np.sqrt(x)):
    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        n = len(Y_hat)
        for Y_start, Y_end in quantizition_intervals_Y:
            for A_start, A_end in constrained_intervals_A:
                y_a_mask = ((Y_start <= Y) & (Y < Y_end) & (A_start <= A) & (A < A_end))
                y_mask = ((Y_start <= Y) & (Y < Y_end))
                cnt_y_a = y_a_mask.sum()
                cnt_y = y_mask.sum()
                sum_y_yhat = (y_mask * Y_hat).sum()
                sum_y_a_yhat = (y_a_mask * Y_hat).sum()
                if cnt_y_a > 0 and cnt_y > 0:
                    curr_nd_loss = torch.abs(sum_y_a_yhat / cnt_y_a - sum_y_yhat / cnt_y) * size_compensation(
                        cnt_y_a / n)
                    nd_losses.append(curr_nd_loss)
        return torch.stack(nd_losses)

    return fairness_metric


def generate_constrained_intervals(num_constrained_intervals):
    endpoints = np.linspace(0, 1, num_constrained_intervals + 1)
    constrained_intervals = []
    for i in range(len(endpoints) - 1):
        constrained_intervals.append((endpoints[i], endpoints[i + 1]))
    return constrained_intervals
