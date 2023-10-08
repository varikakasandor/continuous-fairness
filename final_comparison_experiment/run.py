import numpy as np

from pipeline import FairnessAwareLearningExperiment
from datasets import read_dataset
from syntetic_datasets import read_syntetic
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__ == "__main__":
    dataset_name = "uscensus"

    if dataset_name == "crimes":
        dataset = read_dataset(dataset_name)
        num_constrained_intervals = 2
        intervals = generate_constrained_intervals(num_constrained_intervals)
        beta_metric = generate_beta(intervals, intervals)
        alpha_metric = generate_alpha(intervals, intervals)
        analysis_metric = generate_alpha(intervals, intervals, return_category_names=True)
        num_epochs = 100

    if dataset_name == "uscensus":
        dataset = read_dataset(dataset_name)
        alpha_intervals = generate_constrained_intervals(9)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)
        analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)
        num_epochs = 100

    if dataset_name == "adult":
        dataset = read_dataset(dataset_name)
        alpha_intervals = generate_constrained_intervals(2)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)
        analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)
        num_epochs = 100

    if dataset_name == "syntetic":
        dataset = read_syntetic()
        alpha_intervals = generate_constrained_intervals(2)
        y_intervals = generate_constrained_intervals(2)
        beta_metric = generate_beta(alpha_intervals, y_intervals)
        alpha_metric = generate_alpha(alpha_intervals, y_intervals)
        analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)
        num_epochs = 10

    fairness_weights_beta = fairness_weights_beta = np.logspace(np.log10(0.1), np.log10(30), 30)  # TODO: set it based on eta
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, "Beta", dataset_name, fairness_weights_beta,
                                                      analysis_metric, num_epochs)
    beta_experiment.run_analysis()
    fairness_weights_alpha = np.logspace(np.log10(0.02), np.log10(6), 30)
    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, "Alpha", dataset_name,
                                                       fairness_weights_alpha, analysis_metric, num_epochs)
    alpha_experiment.run_analysis()
