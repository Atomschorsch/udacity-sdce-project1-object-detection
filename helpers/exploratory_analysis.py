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


def show_histogramm(vector, title, axs, **hist_args):
    "Plot histogramm of vector values"
    axs.hist(vector, **hist_args)
    axs.set_title(title)
    # plt.hist(vector, **hist_args)
    # plt.title(title)
    # plt.show()


def get_dataset_size(dataset):
    '''Get the number of elements in a dataset. It actually seems like there is nothing like this currently available'''
    element_count = 0
    for element in dataset:
        element_count += 1
    return element_count


def get_classes_info(dataset):
    '''Get class distribution and label names'''
    classes_list = []  # List of all class occurences
    class_text_vec = []  # Vector of class -> text mapping
    class_count = []  # Vector of class -> count mapping

    def extend_class_text_vec(class_vec, text_vec):
        for idx, element in enumerate(class_vec):
            while len(class_text_vec) <= element:
                class_text_vec.append('')
                class_count.append(0)
            if class_text_vec[element] == '':
                class_text_vec[element] = str(text_vec[idx])
            class_count[element] += 1
            classes_list.append(element)
    for element in dataset:
        extend_class_text_vec(element['classes'], element['classes_text'])
    return classes_list, class_text_vec, class_count


def show_dataset_basics(dataset):
    '''Print dataset main characteristics'''
    # Analysis
    # image count
    # file count
    # width / height distribution
    # box number variance / histogram
    # Show classes
    # class variance / histogram

    num_images = get_dataset_size(dataset)
    widths = []
    heights = []
    num_boxes_per_image = []
    filenames = set()
    for element in dataset:
        filenames.add(element['filename'])
        widths.append(element['width'])
        heights.append(element['height'])
        num_boxes_per_image.append(len(element['boxes']))
    classes_list, class_text_vec, class_count = get_classes_info(dataset)
    num_files = len(filenames)

    # Display information
    print("Dataset info:")
    print(f" {num_images} elements from {num_files} files")
    print(" Classes:")
    for idx, class_text in enumerate(class_text_vec):
        print(f"  {idx}  {class_text}: {class_count[idx]}")

    # Histograms
    # plt.figure()
    # show_histogramm([x['width'] for x in dataset], 'Width distribution')
    # show_histogramm(widths, 'Image width distribution', plt.gca())
    # show_histogramm(heights, 'Image height distribution', plt.gca())
    # show_histogramm(num_boxes_per_image,
    #                 'Number of boxes per image', plt.gca(), bins=max(num_boxes_per_image))
    # show_histogramm(classes_list, 'Class distribution',
    #                 plt.gca(), label=class_text_vec)

    # TODO show all information in one matplotlib subplot
    fig, axs = plt.subplots(2, 2, figsize=(18, 18))
    #axs = axs.reshape(4)
    show_histogramm(widths, 'Image width distribution', axs[0][0])
    show_histogramm(heights, 'Image height distribution', axs[0][1])
    show_histogramm(num_boxes_per_image,
                    'Number of boxes per image', axs[1][0], bins=max(num_boxes_per_image))
    show_histogramm(classes_list, 'Class distribution',
                    axs[1][1], label=class_text_vec, bins=max(classes_list)+1)
    plt.tight_layout()
    plt.show()
    return num_images, class_text_vec, class_count
