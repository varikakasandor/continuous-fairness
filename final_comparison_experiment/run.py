from pipeline import FairnessAwareLearningExperiment
from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__ == "__main__":
    """dataset = read_dataset("crimes")
    num_constrained_intervals = 2
    intervals = generate_constrained_intervals(num_constrained_intervals)
    beta_metric = generate_beta(intervals, intervals)
    alpha_metric = generate_alpha(intervals, intervals)
    analysis_metric = generate_alpha(intervals, intervals, return_category_names=True)"""

    dataset = read_dataset("uscensus")
    alpha_intervals = generate_constrained_intervals(9)
    y_intervals = generate_constrained_intervals(2)
    beta_metric = generate_beta(alpha_intervals, y_intervals)
    alpha_metric = generate_alpha(alpha_intervals, y_intervals)
    analysis_metric = generate_alpha(alpha_intervals, y_intervals, return_category_names=True)

    fairness_weights = [0.01, 0.1, 0.5, 1, 2, 10, 100]

    beta_experiment = FairnessAwareLearningExperiment(dataset, beta_metric, "Beta", "Uscensus", fairness_weights,
                                                      analysis_metric)
    beta_experiment.run_analysis()

    alpha_experiment = FairnessAwareLearningExperiment(dataset, alpha_metric, "Alpha", "Uscensus", fairness_weights,
                                                       analysis_metric)
    alpha_experiment.run_analysis()
