import json
import numpy as np
from collections import namedtuple

ExperimentResults = namedtuple('ExperimentResults',
                         ['y_train', 'a_train', 'y_test', 'a_test', 'categories', 'obj_loss_train', 'nd_loss_train',
                          'bottlenecks_train', 'obj_loss_test', 'nd_loss_test', 'bottlenecks_test', 'fairness_name', 'dataset_name'])


def find_optimal_subplot_dims(num_plots):
    plot_dim_1, plot_dim_2 = int(np.floor(np.sqrt(num_plots))), 1
    while num_plots % plot_dim_1 != 0:
        plot_dim_1 -= 1
    plot_dim_2 = num_plots // plot_dim_1
    return plot_dim_1, plot_dim_2
