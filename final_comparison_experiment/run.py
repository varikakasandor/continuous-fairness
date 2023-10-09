from matplotlib import pyplot as plt
import joblib

from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals
from pipeline import FairnessAwareLearningExperiment
from tools import *

def running_experiments(dataset_name, real_run):
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

    dataset = read_dataset(dataset_name)
    analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)
    num_epochs = 100 if real_run else 1
    num_fairness_weights = 30 if real_run else 3

    fairness_weights_beta = np.logspace(np.log10(0.1), np.log10(25), num_fairness_weights)  # TODO: set it based on eta
    fairness_name = "Beta" if real_run else "Beta_trial"
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, fairness_name, dataset_name, fairness_weights_beta,
                                                      analysis_metric, num_epochs)
    beta_results = beta_experiment.run_analysis()
    joblib.dump(beta_results, f'results/analysis_{fairness_name}_{dataset_name}.joblib')

    fairness_weights_alpha = np.logspace(np.log10(0.02), np.log10(6), num_fairness_weights)
    fairness_name = "Alpha" if real_run else "Alpha_trial"
    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, fairness_name, dataset_name,
                                                       fairness_weights_alpha, analysis_metric, num_epochs)
    alpha_results = alpha_experiment.run_analysis()
    joblib.dump(alpha_results, f'results/analysis_{fairness_name}_{dataset_name}.joblib')
    return alpha_results, beta_results

def load_results(real_run):
    fairness_name = "Beta" if real_run else "Beta_trial"
    beta_results = joblib.load(f'results/analysis_{fairness_name}_{dataset_name}.joblib')
    fairness_name = "Alpha" if real_run else "Alpha_trial"
    alpha_results = joblib.load(f'results/analysis_{fairness_name}_{dataset_name}.joblib')
    return alpha_results, beta_results

if __name__ == "__main__":
    dataset_name = "adult"
    real_run = False
    create_comparison = True
    load_existing_result = True

    if not load_existing_result:
        alpha_results, beta_results = running_experiments(real_run)
    else:
        alpha_results, beta_results = load_results(real_run)


    if create_comparison:
        num_categories = len(beta_results.categories)
        plot_dim_1, plot_dim_2 = find_optimal_subplot_dims(num_categories)
        fig, axes = plt.subplots(plot_dim_1, plot_dim_2, figsize=(plot_dim_1 * 4, plot_dim_2 * 4))

        for i in range(num_categories):
            idx, idy = i // plot_dim_1, i % plot_dim_1
            scatter_beta_train = axes[idx, idy].scatter(beta_results.nd_loss_train[:, i], beta_results.obj_loss_train,
                                                   c=[l[i] for l in beta_results.bottlenecks_train], label='beta_train')
            scatter_alpha_train = axes[idx, idy].scatter(alpha_results.nd_loss_train[:, i], alpha_results.obj_loss_train,
                                                   c=[l[i] for l in alpha_results.bottlenecks_train], label='alpha_train', marker='v')
            scatter_beta_test = axes[idx, idy].scatter(beta_results.nd_loss_test[:, i], beta_results.obj_loss_test,
                                                   c=[l[i] for l in beta_results.bottlenecks_test], label='beta_test')
            scatter_alpha_test = axes[idx, idy].scatter(alpha_results.nd_loss_test[:, i], alpha_results.obj_loss_test,
                                                   c=[l[i] for l in alpha_results.bottlenecks_test], label='alpha_test', marker='v')
            axes[idx, idy].legend(handles=[scatter_beta_train, scatter_alpha_train, scatter_beta_test, scatter_alpha_test])
            axes[idx, idy].set_xlabel('Discrimimnatory loss')
            axes[idx, idy].set_ylabel('Objective loss')
            (Y_start, Y_end), (A_start, A_end) = beta_results.categories[i]
            category_prob_train = ((Y_start < beta_results.y_train) & (beta_results.y_train <= Y_end) & (A_start < beta_results.a_train) & (
                        beta_results.a_train <= A_end)).sum() / len(beta_results.y_train)
            category_prob_test = ((Y_start < beta_results.y_test) & (beta_results.y_test <= Y_end) & (A_start < beta_results.a_test) & (
                        beta_results.a_test <= A_end)).sum() / len(beta_results.y_test)
            category_desc = f"Y: ({Y_start:.3f} - {Y_end:.3f}), A: ({A_start:.3f} - {A_end:.3f}), P_ya_train: {category_prob_train:.3f}, P_ya_test: {category_prob_test:.3f}"
            axes[idx, idy].set_title(category_desc, fontsize="xx-small")

        fig.suptitle(f"{beta_results.fairness_name} vs {alpha_results.fairness_name}")
        plt.tight_layout()
        plt.savefig(f'./plots/comparison_{beta_results.fairness_name}_{alpha_results.fairness_name}_{beta_results.dataset_name}.pdf')
        plt.show()