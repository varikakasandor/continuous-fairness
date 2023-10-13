from matplotlib import pyplot as plt
import joblib
from datetime import datetime
import multiprocessing
import itertools
import pathlib
import copy

from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals
from pipeline import FairnessAwareLearningExperiment
from tools import *


def running_experiments(dataset_name, num_epochs, num_fairness_weights, lr, create_comparison_enabled=True, **kwargs):
    if dataset_name == "crimes":
        intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(intervals, intervals)
        alpha_metric = generate_alpha(intervals, intervals)

    if dataset_name == "uscensus":
        alpha_intervals = generate_constrained_intervals(9)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)

    if dataset_name == "adult":
        alpha_intervals = generate_constrained_intervals(2)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)

    if dataset_name == "synthetic":
        alpha_intervals = generate_constrained_intervals(2)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)

    dataset = read_dataset(dataset_name, **kwargs)
    analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)

    timestamp = datetime.now().timestamp()
    config_str = f"{dataset_name}_{num_epochs}_{lr}_{num_fairness_weights}_{timestamp}"
    if dataset_name == 'synthetic':
        config_str += f'_{"_".join([f"{k}-{v}" for k, v in kwargs.items()])}'

    fairness_weights_beta = np.logspace(np.log10(2), np.log10(15), num_fairness_weights)  # TODO: set it based on eta
    fairness_name = "Beta"
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, f'{fairness_name}_{config_str}',
                                                      dataset_name,
                                                      fairness_weights_beta,
                                                      analysis_metric, lr, num_epochs, external_params=kwargs)
    beta_results = beta_experiment.run_analysis()
    joblib.dump(beta_results, f'results/analysis_{fairness_name}_{config_str}.joblib')

    fairness_weights_alpha = np.logspace(np.log10(0.5), np.log10(3), num_fairness_weights)
    fairness_name = "Alpha"
    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, f'{fairness_name}_{config_str}',
                                                       dataset_name,
                                                       fairness_weights_alpha, analysis_metric, lr, num_epochs,
                                                       external_params=kwargs)
    alpha_results = alpha_experiment.run_analysis()
    joblib.dump(alpha_results, f'results/analysis_{fairness_name}_{config_str}.joblib')
    if create_comparison_enabled:
        create_comparison(alpha_results, beta_results, f"{config_str}_comparison")
    return alpha_results, beta_results


def create_comparison(alpha_results, beta_results, experiment_name):
    num_categories = len(beta_results.categories)
    plot_dim_1, plot_dim_2 = find_optimal_subplot_dims(num_categories)
    fig, axes = plt.subplots(plot_dim_1, plot_dim_2, figsize=(plot_dim_2 * 4, plot_dim_1 * 4))

    for i in range(num_categories):
        idx, idy = i // plot_dim_2, i % plot_dim_2
        scatter_beta_train = axes[idx, idy].scatter(beta_results.nd_loss_train[:, i], beta_results.obj_loss_train,
                                                    c=[l[i] for l in beta_results.bottlenecks_train],
                                                    label='beta_train', marker='x')
        scatter_alpha_train = axes[idx, idy].scatter(alpha_results.nd_loss_train[:, i],
                                                     alpha_results.obj_loss_train,
                                                     c=[l[i] for l in alpha_results.bottlenecks_train],
                                                     label='alpha_train', marker='v')
        scatter_beta_test = axes[idx, idy].scatter(beta_results.nd_loss_test[:, i], beta_results.obj_loss_test,
                                                   c=[l[i] for l in beta_results.bottlenecks_test],
                                                   label='beta_test', marker='x')
        scatter_alpha_test = axes[idx, idy].scatter(alpha_results.nd_loss_test[:, i], alpha_results.obj_loss_test,
                                                    c=[l[i] for l in alpha_results.bottlenecks_test],
                                                    label='alpha_test', marker='v')
        axes[idx, idy].legend(
            handles=[scatter_beta_train, scatter_alpha_train, scatter_beta_test, scatter_alpha_test])
        axes[idx, idy].set_xlabel('Discrimimnatory loss')
        axes[idx, idy].set_ylabel('Objective loss')
        (Y_start, Y_end), (A_start, A_end) = beta_results.categories[i]
        category_prob_train = ((Y_start < beta_results.y_train) & (beta_results.y_train <= Y_end) & (
                A_start < beta_results.a_train) & (
                                       beta_results.a_train <= A_end)).sum() / len(beta_results.y_train)
        category_prob_test = ((Y_start < beta_results.y_test) & (beta_results.y_test <= Y_end) & (
                A_start < beta_results.a_test) & (
                                      beta_results.a_test <= A_end)).sum() / len(beta_results.y_test)
        category_desc = f"Y: ({Y_start:.3f} - {Y_end:.3f}), A: ({A_start:.3f} - {A_end:.3f}), P_ya_train: {category_prob_train:.3f}, P_ya_test: {category_prob_test:.3f}"
        axes[idx, idy].set_title(category_desc, fontsize="xx-small")

    fig.suptitle(f"{beta_results.fairness_name} vs {alpha_results.fairness_name}")
    plt.tight_layout()
    plt.savefig(
        f'./plots/comparison_{experiment_name}.pdf')
    plt.show()


def wrapped_exp(params):
    running_experiments(**params, create_comparison_enabled=True)


if __name__ == "__main__":
    dataset_name = "synthetic"
    real_run = True
    single_run = True
    load_existing_result = False
    use_multiprocessing = False

    if not load_existing_result:
        if single_run:
            # alpha_results, beta_results = running_experiments(dataset_name, 100 if real_run else 2,
            #                                                  20 if real_run else 2,
            #                                                  1e-4, eta=0.4, gamma_0=0.2, gamma_1=0.1,
            #                                                  information_0=0.2, information_1=0.02,
            #                                                  feature_size_0=5, feature_size_1=242,
            #                                                  train_size=6000, test_size=6000) # feature_size_1 should be int(eta * gamma_1 * train_size + 2)
            alpha_results, beta_results = running_experiments(dataset_name, 100 if real_run else 2,
                                                                20 if real_run else 2,
                                                                1e-4, eta=0.4, gamma_0=0.5, gamma_1=0.1,
                                                                information_0=0.2, information_1=0.1,
                                                                feature_size_0=5, feature_size_1=242,
                                                                train_size=6000, test_size=6000) # feature_size_1 should be int(eta * gamma_1 * train_size + 2)

        else:
            default_params = {
                'dataset_name': 'synthetic',
                'num_epochs': 100,
                'num_fairness_weights': 20,
                'train_size': 6000,
                'test_size': 6000,
                'lr': 1e-4,
                'eta': 0.4,
                'gamma_0': 0.2,
                'gamma_1': 0.1,
                'information_0': 0.2,
                'information_1': 0.02,
                'feature_size_0': 10,
                'feature_size_1': 242
            }
            alternative_param_options = {
                'dataset_name': [],
                'num_epochs': [],
                'num_fairness_weights': [],
                'train_size': [],
                'test_size': [],
                'lr': [],
                'eta': [0.1, 0.3, 0.5],
                'gamma_0': [0.01, 0.1, 0.3, 0.5],
                'gamma_1': [0.01, 0.1, 0.3, 0.5],
                'information_0': [0.05, 0.1, 0.5, 1],
                'information_1': [0.0, 0.1, 0.5, 1],
                'feature_size_0': [2, 5, 50],
                'feature_size_1': [5, 30, 60, 100, 300]
            }
            param_combinations = [default_params]
            for (param, options) in alternative_param_options.items():
                for new_val in options:
                    curr_param_combination = copy.deepcopy(default_params)
                    curr_param_combination[param] = new_val
                    param_combinations.append(curr_param_combination)

            if use_multiprocessing:
                num_processes = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=num_processes)
                pool.map(wrapped_exp, param_combinations)
                pool.close()
                pool.join()
            list(map(wrapped_exp, param_combinations))


    else:
        path = pathlib.Path('results/')
        alpha_exps = [filename for filename in path.glob('*.joblib') if 'Alpha' in filename.name]
        beta_exps = [filename for filename in path.glob('*.joblib') if 'Beta' in filename.name]
        experiments = {}
        for filename in path.glob('*.joblib'):
            name = filename.name
            name = name[name.find('_') + 1:]
            fairness_name, name = name[:name.find('_')], name[name.find('_') + 1:]
            name = name.split('.joblib')[0]
            if name not in experiments:
                experiments[name] = [None, None]
            if fairness_name == 'Alpha':
                experiments[name][0] = joblib.load(filename)
            elif fairness_name == 'Beta':
                experiments[name][1] = joblib.load(filename)
            else:
                raise NotImplementedError('Only Alpha and Beta fairness_names are recognized.')

        for experiment_name, (alpha_results, beta_results) in experiments.items():
            if alpha_results is not None and beta_results is not None:
                create_comparison(alpha_results, beta_results, experiment_name)
