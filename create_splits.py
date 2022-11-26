import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger, get_dataset
from sklearn.model_selection import train_test_split


def copy_tf_records(file_list, destination):
    os.makedirs(destination, exist_ok=True)
    for file in file_list:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(destination,file_name))

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    all_tf_records = glob.glob(source + '/*.tfrecord')
    n = len(all_tf_records)
    # dataset = get_dataset(all_tf_records, label_map='label_map.pbtxt')

    test_fac = 0.1
    train_val_fac = 0.2 # percentage of 1-test_fac
    train_train_fac = (1-test_fac)*(1-train_val_fac)

    n_test = round(test_fac*n)
    n_val = round((n-n_test)*train_val_fac)

    random.shuffle(all_tf_records)
    test = all_tf_records[0:n_test]
    val = all_tf_records[n_test:n_test+n_val]
    train = all_tf_records[n_test+n_val:]


    copy_tf_records(test, os.path.join(destination,'test'))
    copy_tf_records(val, os.path.join(destination,'val'))
    copy_tf_records(train, os.path.join(destination,'train'))


    # Alex: open question: Split only the files or also the file content? Each file seems to contain similar data,
    # so each dataset will be quite homogenous?

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=False, default="/mnt/data/processed",
                        help='source data directory')
    parser.add_argument('--destination', required=False, default = "/mnt/data",
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)