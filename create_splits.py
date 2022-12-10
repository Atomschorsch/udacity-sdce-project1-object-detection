import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger, int64_feature, int64_list_feature, \
    bytes_list_feature, bytes_feature, float_list_feature
from sklearn.model_selection import train_test_split
from helpers.project1 import project1_visualize_inspect, parse_record
from helpers.exploratory_analysis import get_dataset_size
import tensorflow as tf


def copy_tf_records(file_list, destination):
    os.makedirs(destination, exist_ok=True)
    for file in file_list:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(destination,file_name))

def ensure_dir_exsists_empty(dir_path):
    """
    Function to ensure a directory is existing and empty
    It will delete existing content without asking
    """
    # Check if folder exists
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def split_files(source, destination, test_fac = 0.1, val_fac = 0.15):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - test_fac [float]: factor, which part of the complete dataset should be used for testing, default 10%
        - val_vac [float]: factor, which part of the complete dataset should be used for validation, default 15%, 0 if no validation is used
    """
    all_tf_records = glob.glob(source + '/*.tfrecord')
    n_files = len(all_tf_records)

    train_path = os.path.join(destination,'train')
    val_path = os.path.join(destination,'val')
    test_path = os.path.join(destination,'test')

    n_test_files = round(test_fac*n_files)
    n_val_files = round(val_fac*n_files)
    n_train_files = round(n_files-n_test_files-n_val_files)
    assert n_files==n_test_files+n_val_files+n_train_files, "File split has not worked properly, sum of datasplits does not sum up to complete dataset."

    random.shuffle(all_tf_records)
    test = all_tf_records[0:n_test_files]
    val = all_tf_records[n_test_files:n_test+n_val_files]
    train = all_tf_records[n_test_files+n_val_files:]    

    copy_tf_records(test, test_path)
    copy_tf_records(val, val_path)
    copy_tf_records(train, train_path)

    print(f"File counts:\n \
        - Train: {len([entry for entry in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, entry))])}\n \
        - Val:  {len([entry for entry in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, entry))])}\n \
        - Test: {len([entry for entry in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, entry))])}")

    return train_path, val_path, test_path

def split_images(source, destination, test_fac = 0.1, val_fac = 0.15):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - test_fac [float]: factor, which part of the complete dataset should be used for testing, default 10%
        - val_vac [float]: factor, which part of the complete dataset should be used for validation, default 15%, 0 if no validation is used
    """
    all_tf_records = glob.glob(source + '/*.tfrecord')
    n_files = len(all_tf_records)

    train_path = os.path.join(destination,'train')
    val_path = os.path.join(destination,'val')
    test_path = os.path.join(destination,'test')

    # Ensure it is shuffled, to get inhomogenous images
    
    full_dataset = tf.data.TFRecordDataset(all_tf_records)
    n_images = get_dataset_size(full_dataset)
    n_test_images = round(test_fac*n_images)
    n_val_images = round(val_fac*n_images)
    n_train_images = round(n_images-n_test_images-n_val_images)
    assert n_images==n_test_images+n_val_images+n_train_images, "Image split has not worked properly, sum of datasplits does not sum up to complete dataset."

    full_dataset = full_dataset.shuffle(buffer_size=n_images)
    train_dataset = full_dataset.take(n_train_images)
    test_dataset = full_dataset.skip(n_train_images)
    val_dataset = test_dataset.skip(n_test_images)
    test_dataset = test_dataset.take(n_test_images)

    ensure_dir_exsists_empty(train_path)
    # Following code for writing is taken from download_process.py
    file_name = os.path.join(train_path,'datafile.tfrecord')
    parsed_train_set = train_dataset.map(parse_record)
    with tf.io.TFRecordWriter(file_name) as writer:
        for idx, data in enumerate(parsed_train_set):
            # we are only saving every 10 frames to reduce the number of similar
            # images. Remove this line if you have enough space to work with full
            # temporal resolution data.
            if idx % 10 == 0:
                # frame = open_dataset.Frame()
                # frame.ParseFromString(bytearray(data.numpy()))
                # encoded_jpeg, annotations = parse_frame(frame)
                # decoded_data = data.map({
                #     'image/encoded': tf.io.FixedLenFeature([], tf.string),
                #     })
                # filename = decoded_data['filename'].replace('.tfrecord', f'_{idx}.tfrecord')
                # tf_example = create_tf_example(filename, decoded_data['image/encoded'], annotations)
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': int64_feature(data['image/height'].numpy()),
                    'image/width': int64_feature(data['image/width'].numpy()),
                    'image/filename': bytes_feature(data['image/filename'].numpy()),
                    'image/source_id': bytes_feature(data['image/source_id'].numpy()),
                    'image/encoded': bytes_feature(data['image/encoded'].numpy()),
                    'image/format': bytes_feature(data['image/format'].numpy()),
                    'image/object/bbox/xmin': float_list_feature(data['image/object/bbox/xmin'].values.numpy()),
                    'image/object/bbox/xmax': float_list_feature(data['image/object/bbox/xmax'].values.numpy()),
                    'image/object/bbox/ymin': float_list_feature(data['image/object/bbox/ymin'].values.numpy()),
                    'image/object/bbox/ymax': float_list_feature(data['image/object/bbox/ymax'].values.numpy()),
                    'image/object/class/text': bytes_list_feature(data['image/object/class/text'].values.numpy()),
                    'image/object/class/label': int64_list_feature(data['image/object/class/label'].values.numpy()),
                }))
                writer.write(tf_example.SerializeToString())

    return train_path, val_path, test_path

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
    # so each dataset will be quite homogenous?

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
    train_path, val_path, test_path = split(args.source, args.destination, test_fac = 0.1, val_fac = 0.15)

    # Visualize different datasets to inspect on homogenities
    project1_visualize_inspect(glob.glob(os.path.join(train_path,'*.tfrecord')))
    project1_visualize_inspect(glob.glob(os.path.join(val_path,'*.tfrecord')))
    project1_visualize_inspect(glob.glob(os.path.join(test_path,'*.tfrecord')))

