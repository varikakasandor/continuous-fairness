import numpy as np

from pipeline import FairnessAwareLearningExperiment
from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__ == "__main__":
    """
    dataset_name = "crimes"
    num_constrained_intervals = 2
    intervals = generate_constrained_intervals(num_constrained_intervals)
    beta_metric = generate_beta(intervals, intervals)
    alpha_metric = generate_alpha(intervals, intervals)
    analysis_metric = generate_alpha(intervals, intervals, return_category_names=True)"""

    dataset_name = "uscensus"
    alpha_intervals = generate_constrained_intervals(9)
    y_intervals = [(-1, 0.5), (0.5, 2)]
    beta_metric = generate_beta(alpha_intervals, y_intervals)
    alpha_metric = generate_alpha(alpha_intervals, y_intervals)
    analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)

    dataset = read_dataset(dataset_name)
    fairness_weights = np.logspace(np.log10(1), np.log10(50), 20)  # TODO: set it based on eta
    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, "Beta_trial", dataset_name, fairness_weights,
                                                      analysis_metric)
    beta_experiment.run_analysis()
    #alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, "Alpha_trial", dataset_name, fairness_weights,
    #                                                   analysis_metric)
    #alpha_experiment.run_analysis()
