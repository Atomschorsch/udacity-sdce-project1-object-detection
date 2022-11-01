#!/usr/bin/python
# Module for exploratory analysis

import tensorflow as tf


def store_tf_record_structure(dataset, n_take=1):
    '''Store dataset structure/content in text file'''
    for idx, sample in enumerate(dataset.take(n_take)):
        # inspect structure
        example = tf.train.Example()
        example.ParseFromString(sample.numpy())
        with open(f"file_content_{idx}.txt", 'w') as f:
            f.write(str(example))


def show_dataset_basics(dataset):
    '''Print dataset main characteristics'''
    print("Dataset Characteristics")
    # image count
    # file count
    # box number variance / histogram
    # Show classes
    # class variance / histogram
