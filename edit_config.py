import argparse
import glob
import os
import re

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def edit(train_dir, eval_dir, batch_size, checkpoint, label_map):
    """
    edit the config file and save it to pipeline_new.config
    args:
    - train_dir [str]: path to train directory
    - eval_dir [str]: path to val OR test directory 
    - batch_size [int]: batch size
    - checkpoint [str]: path to pretrained model
    - label_map [str]: path to labelmap file
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile("pipeline.config", "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    training_files = glob.glob(train_dir + '/*.tfrecord')
    evaluation_files = glob.glob(eval_dir + '/*.tfrecord')

    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    pipeline_config.train_input_reader.label_map_path = label_map
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = training_files

    pipeline_config.eval_input_reader[0].label_map_path = label_map
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = evaluation_files

    config_text = text_format.MessageToString(pipeline_config)
    new_config = "pipeline_new.config"
    with tf.gfile.Open(new_config, "wb") as f:
        f.write(config_text)
    return new_config


def get_current_experiment_id(experiments_folder, prefix='experiment'):
    """Get the id of the latest experiment."""
    all_experiment_subdirs = [d for d in os.listdir(experiments_folder) if os.path.isdir(
        os.path.join(experiments_folder, d)) and prefix in d]
    current_id = -1
    for folder in all_experiment_subdirs:
        reg = re.compile(f'{prefix}(?P<id>\d+)')
        my_match = reg.match(folder)
        current_id = max(int(my_match.group('id')), current_id)
    return current_id


def get_next_experiment_id(experiments_folder, make_dir=True, prefix='experiment'):
    """Get folder and id of next experiment."""
    next_id = get_current_experiment_id(experiments_folder, prefix) + 1
    next_folder = os.path.join(experiments_folder, prefix + str(next_id))
    if make_dir:
        os.mkdir(next_folder)
    return next_folder, next_id


def create_new_config(
    train_dir="/mnt/data/train/",
    eval_dir="/mnt/data/val/",
    batch_size=2,
    checkpoint="experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0",
    label_map="experiments/label_map.pbtxt",
    experiments_folder="experiments"
):
    """Function to create new config"""
    new_config = edit(train_dir, eval_dir, batch_size,
                      checkpoint, label_map)

    # automatically create new experiments folders and copy config there
    # 1. count existing experiments and create new experiment folder
    next_folder, next_id = get_next_experiment_id(experiments_folder)
    # 2. move new config to new folder
    new_config_experiment = os.path.join(next_folder, new_config)
    os.rename(new_config, new_config_experiment)
    return next_folder, new_config_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and process tf files')
    parser.add_argument('--train_dir', required=False, type=str,
                        help='training directory', default="/mnt/data/train/")
    parser.add_argument('--eval_dir', required=False, type=str,
                        help='validation or testing directory', default="/mnt/data/val/")
    parser.add_argument('--batch_size', required=False, type=int,
                        help='number of images in batch', default=2)
    parser.add_argument('--checkpoint', required=False, type=str,
                        help='checkpoint path', default="experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0")
    parser.add_argument('--label_map', required=False, type=str,
                        help='label map path', default="experiments/label_map.pbtxt")
    args = parser.parse_args()
    # new_config = edit(args.train_dir, args.eval_dir, args.batch_size,
    #      args.checkpoint, args.label_map)

    # Alex adaptions
    create_new_config(args.train_dir, args.eval_dir, args.batch_size,
                      args.checkpoint, args.label_map)
