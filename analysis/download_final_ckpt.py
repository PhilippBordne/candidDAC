# script that downloads the final model ckpt from wandb, specify the project
# and the run_id from which to download the ckpt

import wandb
import os
from pathlib import Path

project_name = "CANDID_DAC"
run_id = "pflbflhn"

api = wandb.Api()
run = api.run(f"{project_name}/{run_id}")

model_dir_to_store = str(Path(__file__).parent / f"../results/models/{project_name}/{run_id}")
# change the cwd such that the ckpt is saved in the correct directory
# create the directory if it does not exist
if not os.path.exists(model_dir_to_store):
    os.makedirs(model_dir_to_store)

os.chdir(model_dir_to_store)
print(os.getcwd())
models = []

if run.config['benchmark'] in ["candid_sigmoid", "piecewise_linear"]:
    for i in range(run.config['dim']):
        ckpt_name = f'final_q_network_{i}.pth'
        print(ckpt_name)
        run.file(ckpt_name).download(replace=True)
