#!/usr/bin/python
# Module for exploratory analysis

import tensorflow as tf
import re


def display_structure_of_dataset_item(dataset):
    '''Function to extract structure from tf.train.Example'''
    for sample in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(sample.numpy())
        regex = re.compile(
            "key\: (\"[^\"]+\")\s+value \{\s+(\w+) \{\s+value\:\s+([^\n]{1,150})")
        matches = re.findall(regex, str(example))
        print(f"tf.train.Example structure:")
        for element in matches:
            print(f" - {element[0]} ({element[1]}): {element[2]}")


def show_dataset_basics(dataset):
    '''Print dataset main characteristics'''
    print("Dataset Characteristics")
    # image count
    # file count
    # width / height distribution
    # box number variance / histogram
    # Show classes
    # class variance / histogram
