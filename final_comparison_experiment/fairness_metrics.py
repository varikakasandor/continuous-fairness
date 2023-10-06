import torch
import numpy as np


def generate_alpha(constrained_intervals_A, quantizition_intervals_Y):
    def inside(num, endpoints):
        start, end = endpoints
        return start <= num < end

    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        for inter_Y in quantizition_intervals_Y:
            for inter_A in constrained_intervals_A:
                cnt_y_a = 0
                cnt_y = 0
                sum_y_yhat = torch.tensor(0.0)
                sum_y_a_yhat = torch.tensor(0.0)
                for i in range(len(Y_hat)):  # could be sped up by combining with outer loop
                    if inside(Y[i], inter_Y):
                        cnt_y += 1
                        sum_y_yhat += Y_hat[i]
                        if inside(A[i], inter_A):
                            cnt_y_a += 1
                            sum_y_a_yhat += Y_hat[i]
                if cnt_y_a > 0 and cnt_y > 0:
                    curr_nd_loss = torch.abs(sum_y_a_yhat / cnt_y_a - sum_y_yhat / cnt_y)
                    nd_losses.append(curr_nd_loss)
        return torch.stack(nd_losses)

    return fairness_metric


def generate_beta(constrained_intervals_A, quantizition_intervals_Y, size_compensation=lambda x: np.sqrt(x)):
    def inside(num, endpoints):
        start, end = endpoints
        return start <= num < end

    def fairness_metric(Y_hat, A, Y):
        nd_losses = []
        n = len(Y_hat)
        for inter_Y in quantizition_intervals_Y:
            for inter_A in constrained_intervals_A:
                cnt_y_a = 0
                cnt_y = 0
                sum_y_yhat = torch.tensor(0.0)
                sum_y_a_yhat = torch.tensor(0.0)
                for i in range(len(Y_hat)):  # could be sped up by combining with outer loop
                    if inside(Y[i], inter_Y):
                        cnt_y += 1
                        sum_y_yhat += Y_hat[i]
                        if inside(A[i], inter_A):
                            cnt_y_a += 1
                            sum_y_a_yhat += Y_hat[i]
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
