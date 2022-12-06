#!/usr/bin/python
import os
import subprocess
from edit_config import create_new_config
if __name__ == "__main__":
    # Create new config    
    model_dir , model_config = create_new_config(experiments_folder = 'experiments')

    # Start tensorboard
    subprocess.Popen(['python', '-m', 'tensorboard.main','--logdir', model_dir])

    # Run training
    os.system(f"python experiments/model_main_tf2.py --model_dir={model_dir} --pipeline_config_path={model_config}")