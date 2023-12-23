# Beta Fairness Metric
Every code file is located inside the "final_comparison_experiment" directory. To run the experiments:
```
pipenv shell # Activates the virtual environment
cd final_comparison_experiment # Navigates to the code directory
python run.py # Executes the code
```

### Content of the Repository
 - "data/": Data files are stored here. The code automatically downloads missing files when needed.
 - "plots/": Directory for experiment plots.
 - "records/": Cached experiments (in joblib format) for post-analysis.
 - "datasets.py": Functions for data loading. Each returns the following: x_train, y_train, a_train, x_test, y_test, a_test
   - read_uncensus()
   - read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1)
   - read_adult(nTrain=None, scaler=True, shuffle=False, portion_kept=0.3)
   - read_synthetic_general(etas, gammas, informations, feature_sizes, train_size, test_size)
     - "etas" is an array where the ith element defines the probability of A=i. (The elements must sum to one.)
     - "gammas" is an array where the ith element describes the probability of Y=1 conditioned on A=i.
     - "informations" is an array where the ith element is the scale of the feature vector. (The feature vector contains Y|A=i for each i. Then, uniform [-1,1] noise is added to it, so the larger the informations parameter, the better the model can predict Y.)
     - "train_size" is the size of the training dataset.
     - "test_size" is the size of the testing dataset.
   - read_synthetic(eta, gamma_0, gamma_1, information_0, information_1, feature_size_0, feature_size_1, train_size, test_size, seed=RANDOM_SEED)
     - Identical to read_synthetic_general but with the assumption that num_categories = 2.
 - "fairness_metrics.py": Functions for generating alpha and beta losses.
   - generate_alpha(constrained_intervals_A, quantization_intervals_Y, return_category_names=False)
   - generate_beta(constrained_intervals_A, quantization_intervals_Y, size_compensation=lambda x: np.sqrt(x))
 - "load_records.py": Executable function for post-analysis of records.
   - create_csvs_from_plots()
   - create_all_csvs()
   - create_plots_from_csvs()
 - "models.py": Collection of models.
   - SimpleNN(input_size, num_classes)
 - "pipeline.py": Implements a pipeline for the full lifecycle of the experiments (except post-analysis).
   - MaxLosses: Enum class for alternative maximums. (Recall that fairness is determined by taking the maximum of some metrics for each protected category.)
     - "MAX": The maximum function.
     - "MEAN": The mean function.
     - "MySoftmax": Softmax weighted mean. This is an interpolation between MAX and MEAN.
   - FairnessAwareLearningExperiment(data, fairness_metric, fairness_name, dataset_name, fairness_weights, analysis_metric, lr,
                 num_epochs=100, print_progress=True, external_params={})
     - "data": The data returned by the data_loader functions: (x_train, y_train, a_train, x_test, y_test, a_test).
     - "fairness_metric": Alpha/Beta metric.
     - "fairness_name": The name of the fairness metric for documentation.
     - "dataset_name": The name of the dataset for documentation.
     - "fairness_weights": A list of weights for experiments. The combined loss is the classification loss (BCE) plus the weighted fairness loss.
     - "analysis_metric": Metrics for analysis. (Alpha or Beta)
     - "lr": Learning rate.
     - "num_epochs": The number of epochs for which the model is trained in a single run.
     - "print_progress": Boolean.
     - "external_params": Other parameters that don't affect the experiment but are important for documentation.
 - "run.py": Run file.
   - This is simply a mess...
   - running_experiments(dataset_name, num_epochs, num_fairness_weights, lr, create_comparison_enabled=True, **kwargs)
   - create_comparison(alpha_results, beta_results, experiment_name)
   - wrapped_exp(params) # Wrapped experiment
 - "tools.py": Helper functions.
   - find_optimal_subplot_dims(num_plots) # For creating plots