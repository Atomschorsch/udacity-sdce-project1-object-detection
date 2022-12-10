#!/usr/bin/python
# Module for project1 specific helpers and reusable functions

# Example from https://stackoverflow.com/questions/65783423/tfrecord-print-image-from-tfrecord-file
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import glob
from helpers.visualization import visualize_tf_record_dataset
from helpers.exploratory_analysis import display_structure_of_dataset_item, show_dataset_basics

def parse_record(record):
    '''Function to parse one record.'''
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
        'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
    }
    return tf.io.parse_single_example(record, image_feature_description)


def transform_record(record, box_in_pixel_pos = False):
    """
    Function to transform one record into numpy dict with decoded image
    """
    ret_dict = {
        'image': tf.image.decode_image(record['image/encoded']).numpy(),
        'filename': record['image/filename'].numpy(),
        'width': record['image/width'].numpy(),
        'height': record['image/height'].numpy(),
        'classes_text': record['image/object/class/text'].values.numpy(),
        'classes': record['image/object/class/label'].values.numpy()
    }

    # Handle boxes
    width_factor = 1
    height_factor = 1
    # if boxes in data as [0,1], has to be multiplied with width / height
    if not box_in_pixel_pos:
        width_factor = ret_dict['width']
        height_factor = ret_dict['height']

    boxes_odd = np.array([
        record['image/object/bbox/ymin'].values.numpy()*height_factor,
        record['image/object/bbox/xmin'].values.numpy()*width_factor,
        record['image/object/bbox/ymax'].values.numpy()*height_factor,
        record['image/object/bbox/xmax'].values.numpy()*width_factor,
    ])
    ret_dict['boxes'] = [boxes_odd[:, idx]
                         for idx in range(boxes_odd.shape[1])]
    return ret_dict


def project1_visualize_inspect(tf_record_path_array):
    '''Function to visualize and inspect dataset according to project1'''
    raw_image_dataset = tf.data.TFRecordDataset(tf_record_path_array)
    display_structure_of_dataset_item(raw_image_dataset)

    parsed_image_dataset = raw_image_dataset.map(parse_record)
    # Transform to numpy/python if wanted
    transformed_dataset = [transform_record(
        element) for element in parsed_image_dataset]

    show_dataset_basics(transformed_dataset)

    # Debug Visu
    if False:
        for image_element in transformed_dataset[0:2]:
            plt.imshow(image_element['image'])
            plt.show()
            # display.display(display.Image(data=image_element['image']))

    # Visualize
    visualize_tf_record_dataset(
        transformed_dataset,
        n_show=100,
        x_max=3, y_max=4,
        show_gt_class_names=True,
        class_names=['', 'car', 'pedestrian', '', 'bike'])


if __name__ == "__main__":
    all_tf_records = glob.glob(
        '/mnt/data/processed/*.tfrecord')
    project1_visualize_inspect(all_tf_records)  # all_tf_records[0:5]
    print("End")
