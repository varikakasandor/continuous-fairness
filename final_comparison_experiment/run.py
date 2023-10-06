from pipeline import FairnessAwareLearningExperiment
from datasets import read_dataset
from fairness_metrics import generate_beta, generate_alpha, generate_constrained_intervals

if __name__=="__main__":
    crimes_dataset = read_dataset("crimes")
    num_constrained_intervals = 2
    intervals = generate_constrained_intervals(num_constrained_intervals)
    beta_metric = generate_beta(intervals, intervals)
    analysis_metric = generate_alpha(intervals, intervals)
    fairness_weights = [0.01, 0.1, 0.5, 1, 2, 10, 100]
    beta_experiment = FairnessAwareLearningExperiment(crimes_dataset, beta_metric, fairness_weights, analysis_metric)
    beta_experiment.run_analysis()