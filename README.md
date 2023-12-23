# Beta fairness metric
Every code is located inside "final_comparison_experiment" directory. For running experiments:
```
pipenv shell # To activate the virtual environment
cd final_comparison_experiment # Go to the code directory
python run.py # Run the code.
```

### Content of the reprository
 - "data/": Data files stored here. The code automatically downloads missing files when needed.
 - "plots/": Directory from experiment plots.
 - "recors/": Cached experiments (in joblib format) for post analysis.
 - "datasets.py": Data loading functions. Each returns the following: x_train, y_train, a_train, x_test, y_test, a_test
   - read_uncensus()
   - read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1)
   - read_adult(nTrain=None, scaler=True, shuffle=False, portion_kept=0.3)
   - read_synthetic_general(etas, gammas, informations, feature_sizes, train_size, test_size)
     - "etas" is an array where the ith element defines the probability of A=i. (The elements must sum to one.)
     - "gammas" is an array where the ith element describes the probability of Y=1 conditioned on A=i.
     - "informations" a is an array where the ith element is scale of the feature vector. (The feature vector contains the Y|A=i for each i. Then later a unifor [-1,1] noise added to it, so the larger the informations parameter the better the model can predict Y.)
     - "train_size" is the size of the train_dataset.
     - "test_size" is the size of the test_dataset.
   - read_synthetic(eta, gamma_0, gamma_1, information_0, information_1, feature_size_0, feature_size_1, train_size, test_size, seed=RANDOM_SEED)
     - Identical to read_synthetic_general but with the assumption that num_categories = 2.
 - "fairness_metircs.py": Function for generating alpha and beta losses.
   - generate_alpha(constrained_intervals_A, quantizition_intervals_Y, return_category_names = False)
   - generate_beta(constrained_intervals_A, quantizition_intervals_Y, size_compensation=lambda x: np.sqrt(x))
 - "load_records.py": Executable function for post analysing records.
   - create_csvs_from_plots()
   - create_all_csvs()
   - create_plots_from_csvs()
 - "models.py": Model collection.
   - SimpleNN(input_size, num_classes)
 - "pipeline.py": Implements a pipeline for the experiments full lifecycle (exept post analysis).
   - MaxLosses: Enum class for alternative maximums. (Recall that fairness is determined by taking the maximum of some metrics for each protected category.)
     - "MAX": The maximum function.
     - "MEAN": The mean function.
     - "MySoftmax": Softmax weighted mean. This is an interpolation between MAX and MEAN.
   - FairnessAwereLearningExperiment(data, fairness_metric, fairness_name, dataset_name, fairness_weights, analysis_metric, lr,
                 num_epochs=100, print_progress=True, external_params={})
     - "data": The data returned by the data_loader functions: (x_train, y_train, a_train, x_test, y_test, a_test).
     - "fairness_metric": Alpha/Beta metric.
     - "fairness_name": The name of the fairness_metrics for documentation.
     - "dataset_name": The name of the dataset for documentation.
     - "fairness_weights": A list of weights for experiments. The combined loss is the classification loss (BCE) plus the weigthed fairness loss.
     - "analysis_metric": Metrics for analysis. (Alpha or Beta)
     - "lr": Learning rate.
     - "num_epochs": The number of epochs the model trained for in a single run.
     - "print_progress": Bool.
     - "external_params": Other parameters that don't affect the experiment but important for the documentation.
 - "run.py": Run file.
   - This is simply a mess...
   - running_experiments(dataset_name, num_epochs, num_fairness_weights, lr, create_comparison_enabled=True, **kwargs)
   - create_comparison(alpha_results, beta_results, experiment_name)
   - wrapped_exp(params) # Wrapped experiment
 - "tools.py": Helper functions.
   - find_optimal_subplot_dims(num_plots) # For creating plots