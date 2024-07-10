# CANDID DAC: Leveraging Coupled Action Dimensions with Importance Differences in DAC

## Overview
This work addresses challenges in dynamic algorithm configuration (DAC) by simulating high-dimensional action spaces with interdependencies and varying importance between action dimensions. 
We propose sequential policies to effectively manage these properties, significantly outperforming independent learning of factorized policies and overcoming scalability limitations. 
Read the full paper at *tbd*.

## Repository Structure
- **DACBench/**: Contains our fork of DACBench extended by the Piecewise Linear benchmark, we will replace this by a git submodule after double-blind review. 
- **analysis/**: Contains scripts and notebooks for extracting data from wandb and for generating plots.
- **scripts/**: Main script tp run all our experiments and conf subdirectories to setup experiments with hydra.
- **src/candid_dac/**: Implementation of algorithms and policies to evaluat on Piecewise Linear benchmark.
- **setup.py**: Script for installing the package and its dependencies.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Conda

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/PhilippBordne/candidDAC.git
   cd candidDAC
2. Get the DACBench submodule, which contains the benchmarks:
   ```sh
   git submodule update --init --recursive
2. Create and activate the conda environment with `python=3.10`:
   ```sh
   conda env create -n candid python=3.10
   conda activate candid
3. Install the package (make sure `DACBench/` is at root):
   ```sh
   pip install -e .
### Reproducing the Experiments
 > **Note:** We use wandb to track all the metrics in our experiments. Otherwise in our current implementation we don't log a metric but only print the training reward to the console. <br>
 To track metrics please specify a wandb project to plot to in the hydra config you plan to run (under `scripts/conf`)

To reproduce the experiments, follow these steps:
1. Navigate to the scripts directory:
   ```sh
   cd scripts
2. To run a simple example of the SAQL algorithm on the piecewise_linear_2d benchmark, use the following command:
   ```sh
   python dqn_factorized_policies.py --config-name=simple_example
         
3.  To reproduce a specific experiment you can select which algorithm to run on which benchmark setup. You can also select the hyperparameters to use for the run but we always used best_<algorith>.yaml. You will also have to specify the seed to run the experiment on. This example runs SAQL on the 10D Piecewise Linear benchmark using seed 0: 
    ```sh
    python dqn_factorized_policies.py --config-name=config +benchmark=piecewise_linear_10d +algorithm=saql +hyperparameters=best_saql +seed=0
    ```
    Replace **saql, piecewise_linear_10d,** and **best_saql** with the algorithm, benchmark, and hyperparameters of your choice.
 
5. To do a **sweep over different hyperparameter settings** you can sample a random configuration from the spaces as specified in `scripts/dqn_factorized_policies.py`: To do so, run the following command:
   ```sh
   python dqn_factorized_policies.py --config-name=config +benchmark=sigmoid_5d +algorithm=saql +hyperparameters=random_config +hyperparameters.seed=123 +seed=321
   ```
    This will sample a random configuration using the seed 123 (and identify as config_id 123 in you wandb project). The seed 321 is used to run the experiment with the sampled configuration (and identified as seed in your wandb project).
    We note that we used the 5D Sigmoid benchmark to identify our hyperparameters.
   
### Configuration Files

 The configuration files for the experiments are located in the `scripts/conf/` directory. These files contain settings for experiment design choices **benchmark, algorithm, hyperparameter configuration**.

### Results
 Results from the experiments, including logs and output data, will be stored in the **results/** directory. You can analyze these results using the scripts and notebooks provided in the **analysis/** directory. We provide the configurations and metrics for the experiments presented in the paper in `analysis/run_data/`.

### Acknowledgements
