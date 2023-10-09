import numpy as np

from pipeline import FairnessAwareLearningExperiment
from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__ == "__main__":
    dataset_name = "adult"
    real_run = True

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
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, "Beta" if real_run else "Beta_trial", dataset_name, fairness_weights_beta,
                                                      analysis_metric, num_epochs)
    beta_results = beta_experiment.run_analysis()

    fairness_weights_alpha = np.logspace(np.log10(0.02), np.log10(6), num_fairness_weights)
    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, "Alpha" if real_run else "Alpha_trial", dataset_name,
                                                       fairness_weights_alpha, analysis_metric, num_epochs)
    alpha_results = alpha_experiment.run_analysis()
