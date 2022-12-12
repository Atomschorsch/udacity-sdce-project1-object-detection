#!/usr/bin/python
import os
import subprocess
from edit_config import create_new_config
if __name__ == "__main__":
    # Create new config    
    model_dir , model_config = create_new_config(experiments_folder = 'experiments')

    # Start tensorboard
    # python -m tensorboard.main --logdir experiments
    subprocess.Popen(['python', '-m', 'tensorboard.main','--logdir', model_dir])


    # Run training (~20 min/25000 steps on RTX4090)
    # Reference training: python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
    # Experiment training: python experiments/model_main_tf2.py --model_dir=experiments/experiment0/ --pipeline_config_path=experiments/experiment0/pipeline_new.config
    os.system(f"python experiments/model_main_tf2.py --model_dir={model_dir} --pipeline_config_path={model_config}")

    # Evaluation
    # python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
    # python experiments/model_main_tf2.py --model_dir=experiments/experiment0/ --pipeline_config_path=experiments/experiment0/pipeline_new.config --checkpoint_dir=experiments/experiment0/

    # Export
    # python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/experiment10/pipeline_new.config --trained_checkpoint_dir experiments/experiment10/ --output_directory experiments/experiment10/exported/