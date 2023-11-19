from collections import namedtuple

import numpy as np

ExperimentResults = namedtuple('ExperimentResults',
                               ['y_train', 'a_train', 'y_test', 'a_test', 'categories', 'obj_loss_train',
                                'nd_loss_train', 'per_category_loss_train',
                                'bottlenecks_train', 'obj_loss_test', 'nd_loss_test', 'per_category_loss_test',
                                'bottlenecks_test', 'fairness_name', 'dataset_name'])


def find_optimal_subplot_dims(num_plots):
    plot_dim_1, plot_dim_2 = int(np.floor(np.sqrt(num_plots))), 1
    while num_plots % plot_dim_1 != 0:
        plot_dim_1 -= 1
    plot_dim_2 = num_plots // plot_dim_1
    return plot_dim_1, plot_dim_2


def synchronise_ya_in_dataset(x_train, y_train, a_train, x_test, y_test, a_test):
    unique_train_pairs = set(zip(y_train, a_train))
    unique_test_pairs = set(zip(y_test, a_test))
    common_pairs = list(unique_train_pairs.intersection(unique_test_pairs))
    train_mask = np.array([pair in common_pairs for pair in zip(y_train, a_train)])
    test_mask = np.array([pair in common_pairs for pair in zip(y_test, a_test)])
    return x_train[train_mask], y_train[train_mask], a_train[train_mask], x_test[test_mask], y_test[test_mask], a_test[test_mask]


RANDOM_SEED = 42
MARKER_STRONG_WIDTH = 4
