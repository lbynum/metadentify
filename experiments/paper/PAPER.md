# Paper examples

Below are additional usage examples for experiments from the paper. In the examples below, small task numbers and dataset sizes are used to demonstrate local runs. See Appendix B in the paper for exact parameter settings used in various runs. More general setup and installation instructions can be found in `README.md`.

### Optimal adjustment set examples

```bash
# nn-based scm function class
uv run run-experiment --experiment_setup_path adjustment.py --experiment_name optimal_adjustment_both --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --scm_function_class nn 

# bernoulli scm function class
uv run run-experiment --experiment_setup_path adjustment.py --experiment_name optimal_adjustment_both --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --scm_function_class binary
```

### Transportability examples

```bash
# interventional data + hidden confounding
uv run run-experiment --experiment_setup_path transportability.py --experiment_name interventional_mixture --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --frac_interventional 0.2 --hidden_confounder_weight 0.8
```

### Counterfactual examples

```bash
# paired interventional + empirical data for CATE
uv run run-experiment --experiment_setup_path counterfactual.py --experiment_name cf_vs_int --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --query_name CATE --frac_paired_interventional 1
```

```bash
# paired counterfactual + empirical data for ITE
uv run run-experiment --experiment_setup_path counterfactual.py --experiment_name cf_vs_int --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --query_name ITE --frac_paired_counterfactual 1
```

### Appendix examples 

```bash
# proximal case
uv run run-experiment --experiment_setup_path linear.py --experiment_name proxy --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --dataset_size 1000

# instrument case
uv run run-experiment --experiment_setup_path linear.py --experiment_name iv --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --dataset_size 1000

# known confounder case
uv run run-experiment --experiment_setup_path linear.py --experiment_name confounder --num_train_tasks 1000 --num_val_tasks 100 --num_test_tasks 100 --max_epochs 100 --dataset_size 1000
```



