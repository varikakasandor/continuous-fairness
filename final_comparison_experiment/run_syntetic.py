import numpy as np

from pipeline import FairnessAwareLearningExperiment
from syntetic_datasets import read_syntetic
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__ == "__main__":
    dataset = read_syntetic()
    alpha_intervals = generate_constrained_intervals(2)
    y_intervals = generate_constrained_intervals(2)
    beta_metric = generate_beta(alpha_intervals, y_intervals)
    alpha_metric = generate_alpha(alpha_intervals, y_intervals)
    analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)

    fairness_weights = np.logspace(np.log10(2), np.log10(300), 20)

    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, "Beta_trial", "Syntetic", fairness_weights,
                                                      analysis_metric)
    beta_experiment.run_analysis()

    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, "Alpha_trial", "Syntetic", fairness_weights,
                                                       analysis_metric)
    alpha_experiment.run_analysis()