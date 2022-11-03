#!/usr/bin/python
# Module for exploratory analysis

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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


def show_histogramm(vector, title, **hist_args):
    "Plot histogramm of vector values"
    plt.hist(vector, **hist_args)
    plt.title(title)
    plt.show()


def show_dataset_basics(dataset):
    '''Print dataset main characteristics'''
    print("Dataset Characteristics")

    show_histogramm([x['width'] for x in dataset], 'Widths')
    show_histogramm([x['height'] for x in dataset], 'Heights')
    num_boxes_per_image = [len(x['boxes']) for x in dataset]
    show_histogramm(num_boxes_per_image,
                    'Number of boxes per image', bins=max(num_boxes_per_image))

    # image count
    # file count
    # width / height distribution
    # box number variance / histogram
    # Show classes
    # class variance / histogram
    print("End")
