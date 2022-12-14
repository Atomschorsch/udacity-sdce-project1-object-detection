#!/usr/bin/python
# Module for splitting tooling

import glob
import os
import random
import shutil

from helpers.exploratory_analysis import get_dataset_size
import tensorflow as tf


def copy_tf_records(file_list, destination):
    """
    Function to copy a filelist to a certain directory
    """
    os.makedirs(destination, exist_ok=True)
    for file in file_list:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(destination, file_name))


def ensure_dir_exsists_empty(dir_path):
    """
    Function to ensure a directory is existing and empty
    It will delete existing content without asking
    """
    # Check if folder exists
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def write_dataset_to_file(dataset, directory, filename='set', split=1):
    """
    Function to write a dataset to 1 or n different files
    (Split not yet implemented)
    """
    file_name = os.path.join(directory, f'{filename}.tfrecord')
    with tf.io.TFRecordWriter(file_name) as writer:
        for example in dataset:
            writer.write(example.numpy())


def split_files(source, destination, test_fac=0.1, val_fac=0.15):
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

    train_path = os.path.join(destination, 'train')
    val_path = os.path.join(destination, 'val')
    test_path = os.path.join(destination, 'test')

    n_test_files = round(test_fac * n_files)
    n_val_files = round(val_fac * n_files)
    n_train_files = round(n_files - n_test_files - n_val_files)
    assert n_files == n_test_files + n_val_files + \
        n_train_files, "File split has not worked properly, sum of datasplits does not sum up to complete dataset."

    random.shuffle(all_tf_records)
    test = all_tf_records[0:n_test_files]
    val = all_tf_records[n_test_files:n_test_files + n_val_files]
    train = all_tf_records[n_test_files + n_val_files:]

    copy_tf_records(test, test_path)
    copy_tf_records(val, val_path)
    copy_tf_records(train, train_path)

    print(f"File counts:\n \
        - Train: {len([entry for entry in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, entry))])}\n \
        - Val:  {len([entry for entry in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, entry))])}\n \
        - Test: {len([entry for entry in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, entry))])}")

    return train_path, val_path, test_path


def split_images(source, destination, test_fac=0.1, val_fac=0.15):
    """
    Create three splits from given processed records in source folders.
    Contented images will be collected, shuffled, and stored again in three different datasets (tfrecord files) with train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - test_fac [float]: factor, which part of the complete dataset should be used for testing, default 10%
        - val_vac [float]: factor, which part of the complete dataset should be used for validation, default 15%, 0 if no validation is used
    """
    all_tf_records = glob.glob(source + '/*.tfrecord')
    n_files = len(all_tf_records)

    train_path = os.path.join(destination, 'train')
    val_path = os.path.join(destination, 'val')
    test_path = os.path.join(destination, 'test')

    # Ensure it is shuffled, to get inhomogenous images

    full_dataset = tf.data.TFRecordDataset(all_tf_records)
    n_images = get_dataset_size(full_dataset)
    n_test_images = round(test_fac * n_images)
    n_val_images = round(val_fac * n_images)
    n_train_images = round(n_images - n_test_images - n_val_images)
    assert n_images == n_test_images + n_val_images + \
        n_train_images, "Image split has not worked properly, sum of datasplits does not sum up to complete dataset."

    full_dataset = full_dataset.shuffle(buffer_size=n_images)
    train_dataset = full_dataset.take(n_train_images)
    test_dataset = full_dataset.skip(n_train_images)
    val_dataset = test_dataset.skip(n_test_images)
    test_dataset = test_dataset.take(n_test_images)

    # Write splitted data
    ensure_dir_exsists_empty(train_path)
    write_dataset_to_file(train_dataset, train_path, filename='train_set')
    ensure_dir_exsists_empty(val_path)
    write_dataset_to_file(val_dataset, val_path, filename='val_set')
    ensure_dir_exsists_empty(test_path)
    write_dataset_to_file(test_dataset, test_path, filename='test_set')

    return train_path, val_path, test_path
