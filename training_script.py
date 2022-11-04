import os

# Print current dir

import argparse

# When running parallel jobs, we can for some reason run into issues if
# the processes are not started from within the main scope

def do_training():
    data_dir = "wavs11025"
    log_dir = "log_11025"
    model_dir = "model_11025"
    
    training_args = {
        "model_dir": model_dir,
        "data_dirs": [data_dir],
        "log_dir": log_dir,
        "max_steps": None,
        "fp16": False,
        "wandb_log": True,
        "project_name": "diffwave_gpu_1.0",
    }

    # os.system(f"python -m diffwave.preprocess {directory}")

    # os.system(f"python -m diffwave model_11025 {directory}")

    args = argparse.Namespace(**training_args)

    if not os.path.exists(data_dir):
        print(f"Can't find data directory at {data_dir}, make sure it exists and is being mounted correctly")
        return

    # Check if directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    

    from diffwave.learner import train
    from diffwave.params import params

    train(args, params)


if __name__ == "__main__":
    do_training()
