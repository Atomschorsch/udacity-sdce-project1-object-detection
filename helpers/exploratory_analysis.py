#!/usr/bin/python
# Module for exploratory analysis
import matplotlib
# Necessary to see matplotlib outside of container
import tkinter
matplotlib.use('TKAgg')
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


def show_histogramm(ax, vector, title, show_values=True, labels=[], ** hist_args):
    "Plot histogramm of vector values"
    counts, bins, patches = ax.hist(vector, **hist_args)
    # ax.set_xticks(bins)
    ax.set_title(title)
    # Following code to show hist values in diagram taken from
    # https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
    if show_values:
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            # Label the raw counts
            ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

            # Label the percentages
            percent = '%0.0f%%' % (100 * float(count) / counts.sum())
            ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    # own code
    if labels:
        for idx, count in enumerate(counts):
            ax.annotate(str(labels[idx]), (bins[idx], count))
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
    # show_histogramm(plt.gca(), [x['width'] for x in dataset], 'Width distribution')

    # Show all information in one matplotlib subplot
    fig, axs = plt.subplots(2, 2, figsize=(18, 18), tight_layout=True)
    fig.suptitle(
        f"Dataset ({num_images} images)\nclasses {class_text_vec}", fontsize=14)
    show_histogramm(axs[0][0], widths, 'Image width distribution', )
    show_histogramm(axs[0][1], heights, 'Image height distribution')
    show_histogramm(axs[1][0], num_boxes_per_image,
                    'Number of boxes per image', show_values=False, bins=max(num_boxes_per_image))
    show_histogramm(axs[1][1], classes_list, 'Class distribution', labels=class_text_vec,
                    bins=[idx - 0.5 for idx in range(0, len(class_text_vec) + 1)])
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    return num_images, class_text_vec, class_count
