from LangBoltz_backbone import run_experiment as run_experiment_backbone
from LangBoltz_sidechain_MLM import run_experiment as run_experiment_sidechain_MLM
from LangBoltz_sidechain_ARM import run_experiment as run_experiment_sidechain_ARM
import mdtraj as md
import time
import torch
import shutil
from tqdm import tqdm
import json
import yaml
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
def main(config):
    num_sample = config["num_sample"]
    batch_size = config["batch_size"]
    sequence = config["sequence"]
    protein_name = config["protein_name"]
    sidechain_ARM = config["sidechain_ARM"]
    run_experiment_backbone(
        num_sample=num_sample,
        batch_size=batch_size,
        sequence=sequence,
        protein_name=protein_name,
    )

    if sidechain_ARM:
        run_experiment_sidechain_ARM(protein_name=protein_name)
    else:
        run_experiment_sidechain(protein_name=protein_name)
    shutil.rmtree("stepone_merge", ignore_errors=True)
    shutil.rmtree("LangBoltz_ensemble", ignore_errors=True)

if __name__ == "__main__":
    config = load_config("config.yml")
    main(config)

        