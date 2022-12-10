#!/usr/bin/python
import argparse
import glob
import os

from utils import get_module_logger
from helpers.project1 import project1_visualize_inspect
from helpers.split import split_images, split_files



def split(source, destination, test_fac = 0.1, val_fac = 0.15):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - test_fac [float]: factor, which part of the complete dataset should be used for testing, default 10%
        - val_vac [float]: factor, which part of the complete dataset should be used for validation, default 15%, 0 if no validation is used
    """
    # Alex: open question: Split only the files or also the file content? Each file seems to contain similar data (e.g. all files from one recording drive),
    # so each dataset will be quite homogenous? See argumentation in writeup

    # Implemented both functions
    # train_path, val_path, test_path = split_files(source, destination, test_fac = 0.1, val_fac = 0.15)
    train_path, val_path, test_path = split_images(source, destination, test_fac = 0.1, val_fac = 0.15)
    return train_path, val_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=False, default="/mnt/data/processed",
                        help='source data directory')
    parser.add_argument('--destination', required=False, default = "/mnt/data",
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)

    logger.info('Creating splits...')
    # TUTOR split function temporarily deactivated to just view the datasets and not recreate
    # train_path, val_path, test_path = split(args.source, args.destination, test_fac = 0.1, val_fac = 0.15)
    train_path = os.path.join(args.destination, 'train')
    val_path = os.path.join(args.destination, 'val')
    test_path = os.path.join(args.destination, 'test')

    # Visualize different datasets to inspect on homogenities
    logger.info('Inspecting splits...')
    project1_visualize_inspect(glob.glob(os.path.join(train_path,'*.tfrecord')))
    project1_visualize_inspect(glob.glob(os.path.join(val_path,'*.tfrecord')))
    project1_visualize_inspect(glob.glob(os.path.join(test_path,'*.tfrecord')))

