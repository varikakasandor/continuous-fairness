from matplotlib import pyplot as plt
import joblib
from datetime import datetime
import multiprocessing
import itertools
import pathlib

from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals
from pipeline import FairnessAwareLearningExperiment
from final_comparison_experiment.tools import *


def running_experiments(dataset_name, num_epochs, num_fairness_weights, lr, create_comparison_enabled = True, **kwargs):
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

    fairness_weights_beta = np.logspace(np.log10(0.1), np.log10(25), num_fairness_weights)  # TODO: set it based on eta
    fairness_name = "Beta"
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, f'{fairness_name}_{config_str}',
                                                      dataset_name,
                                                      fairness_weights_beta,
                                                      analysis_metric, lr, num_epochs)
    beta_results = beta_experiment.run_analysis()
    joblib.dump(beta_results, f'results/analysis_{fairness_name}_{config_str}.joblib')

    fairness_weights_alpha = np.logspace(np.log10(0.02), np.log10(6), num_fairness_weights)
    fairness_name = "Alpha"
    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, f'{fairness_name}_{config_str}',
                                                       dataset_name,
                                                       fairness_weights_alpha, analysis_metric, lr, num_epochs)
    alpha_results = alpha_experiment.run_analysis()
    joblib.dump(alpha_results, f'results/analysis_{fairness_name}_{config_str}.joblib')
    if create_comparison_enabled:
        create_comparison(alpha_results, beta_results, f"{config_str}_comparison")
    return alpha_results, beta_results


def create_comparison(alpha_results, beta_results, experiment_name):
    num_categories = len(beta_results.categories)
    plot_dim_1, plot_dim_2 = find_optimal_subplot_dims(num_categories)
    fig, axes = plt.subplots(plot_dim_1, plot_dim_2, figsize=(plot_dim_1 * 4, plot_dim_2 * 4))

    for i in range(num_categories):
        idx, idy = i // plot_dim_1, i % plot_dim_1
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
    running_experiments(**params, create_comparison_enabled=False)


if __name__ == "__main__":
<<<<<<< HEAD
    dataset_name = "synthetic"
    real_run = False
    load_existing_result = False
=======
    dataset_name = "adult"
    real_run = True
    single_run = True
    load_existing_result = False
    use_multiprocessing = False
>>>>>>> ce68454c86ace5f968d6d8b6d9d9b4d1cd968b3d

    if not load_existing_result:
        if single_run:
            alpha_results, beta_results = running_experiments(dataset_name, 350 if real_run else 2, 20 if real_run else 2, 1e-5)
        else:
            default_params = {
                'dataset_name': dataset_name,
                'num_epochs': 350,
                'num_fairness_weights': 26,
            }
            # param_combinations = [{**(default_params.copy()), **({'lr': lr, 'eta': eta, 'gamma_0': gamma_0, 'gamma_1': gamma_1})} for (lr, eta, (gamma_0, gamma_1)) in itertools.product([1e-5, 3e-5, 1e-4, 3e-6], np.linspace(0.01, 0.5, 6), [(0.1, 0.2), (0.3, 0.3), (0.1, 0.1), (0.1, 0.5)])]
            param_combinations = [
                {**(default_params.copy()), **({'lr': lr, 'eta': eta, 'gamma_0': gamma_0, 'gamma_1': gamma_1})} for
                (lr, eta, (gamma_0, gamma_1)) in itertools.product([1e-5, 1e-4], np.linspace(0.01, 0.5, 6), [(0.2, 0.1)])]
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
