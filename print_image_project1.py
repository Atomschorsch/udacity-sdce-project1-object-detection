# Example from https://stackoverflow.com/questions/65783423/tfrecord-print-image-from-tfrecord-file
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import IPython.display as display
import numpy as np
import glob
from helpers.visualization import visualize_tf_record_dataset
from helpers.exploratory_analysis import store_tf_record_structure, show_dataset_basics


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


def decode_record(record):
    ret_dict = {
        'image': tf.image.decode_image(record['image/encoded']).numpy(),
        'filename': record['image/filename'].numpy(),
        'width': record['image/width'].numpy(),
        'height': record['image/height'].numpy(),
        'classes_text': record['image/object/class/text'].values.numpy(),
        'classes': record['image/object/class/label'].values.numpy()
    }
    # boxes in data as [0,1], has to be multiplied with width / height
    boxes_odd = np.array([
        record['image/object/bbox/ymin'].values.numpy()*ret_dict['height'],
        record['image/object/bbox/xmin'].values.numpy()*ret_dict['width'],
        record['image/object/bbox/ymax'].values.numpy()*ret_dict['height'],
        record['image/object/bbox/xmax'].values.numpy()*ret_dict['width'],
    ])
    ret_dict['boxes'] = [boxes_odd[:, idx]
                         for idx in range(boxes_odd.shape[1])]
    return ret_dict


def project1_visualize_inspect(tf_record_path_array):
    '''Function to visualize and inspect dataset according to project1'''
    raw_image_dataset = tf.data.TFRecordDataset(tf_record_path_array)
    store_tf_record_structure(raw_image_dataset)
    show_dataset_basics(raw_image_dataset)

    parsed_image_dataset = raw_image_dataset.map(parse_record)

    # Debug Visu
    if False:
        for image_features in parsed_image_dataset.take(3):
            image_raw = image_features['image/encoded'].numpy()
            plt.imshow(tf.image.decode_image(image_raw).numpy())
            plt.show()
            # display.display(display.Image(data=image_raw))

    # Visualization & inspection
    for element in parsed_image_dataset.take(10):
        decoded_element = decode_record(element)

    # Visualize
    visualize_tf_record_dataset(
        parsed_image_dataset,
        x_max=4, y_max=4,
        show_gt_class_names=True,
        class_names=['', 'car', '', 'pedestrian', 'bike'],
        decode_fun=decode_record)


if __name__ == "__main__":
    all_tf_records = glob.glob(
        'C:\\Repos\\Udacity\\project1\\data\\processed\\*.tfrecord')
    sample_path = [all_tf_records[0]]
    project1_visualize_inspect(all_tf_records[0:4])
    # input("Press Enter to continue...")
