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
        # tf.train.FloatList(value=value)
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
        'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
    }
    return tf.io.parse_single_example(record, image_feature_description)


def from_example(tf_record_path_array):
    '''Function example from https://www.tensorflow.org/tutorials/load_data/tfrecord'''
    raw_image_dataset = tf.data.TFRecordDataset(tf_record_path_array)
    store_tf_record_structure(raw_image_dataset)
    show_dataset_basics(raw_image_dataset)

    # # Create a dictionary describing the features.
    # image_feature_description = {
    #   'image/height': tf.io.FixedLenFeature([], tf.int64),
    #   'image/width': tf.io.FixedLenFeature([], tf.int64),
    #   'image/filename': tf.io.FixedLenFeature([], tf.string),
    #   'image/source_id': tf.io.FixedLenFeature([], tf.string),
    #   'image/encoded': tf.io.FixedLenFeature([], tf.string),
    #   'image/format': tf.io.FixedLenFeature([], tf.string),
    #   'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),# tf.train.FloatList(value=value)
    #   'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
    #   'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
    #   'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
    #   'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
    #   'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
    #   }

    # def _parse_image_function(example_proto):
    #   # Parse the input tf.train.Example proto using the dictionary above.
    #   return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(parse_record)
    parsed_image_dataset

    # Debug Visu
    if False:
        for image_features in parsed_image_dataset.take(3):
        image_raw = image_features['image/encoded'].numpy()
        box_x_mins = image_features['image/object/bbox/xmin'].values

        tf.image.decode_image(image_raw)
        plt.imshow(tf.image.decode_image(image_raw).numpy())
        plt.show()
        # display.display(display.Image(data=image_raw))

    # Visualize
    visualize_tf_record_dataset(parsed_image_dataset, x_max=3, y_max=4)


def read_plot_tf_records(tf_record_path):
    print("read_plot_tf_records")
    data = tf.data.TFRecordDataset(tf_record_path)
    if not os.path.exists(tf_record_path):
        raise Exception(f"Path {tf_record_path} does not exist!")
    for idx, sample in enumerate(data.take(3)):
        # inspect structure
        example = tf.train.Example()
        example.ParseFromString(sample.numpy())
        with open(f"file_content_{idx}.txt", 'w') as f:
            f.write(str(example))
        # convert
        feature, label = parse_tfr_element(sample)
    print("End")


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        # 'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float64),
        # 'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float64),
        # 'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float64),
        # 'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float64),
        # 'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
        # 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    content = tf.io.parse_single_example(element, data)

    height = content['image/height']
    print(height)
    width = content['image/width']
    print(width)
    filename = content['image/filename']
    print(filename)
    label = content['image/object/class/label']
    raw_image = content['image/encoded']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.int16)
    feature = tf.reshape(feature, shape=[height, width, 3])
    return (feature, label)


def get_dataset_small(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )
    return dataset


def run_dataset_example(tfrecord_file, num_records_to_plot=3):
    # tfrecord_file = './data/small_images.tfrecords'
    dataset = get_dataset_small(tfrecord_file)

    for sample in dataset.take(num_records_to_plot):
        print(f"Image shape: {sample[0].shape}, label: {sample[1]}")
        plt.imshow(sample[0])
        plt.show()


if __name__ == "__main__":
    all_tf_records = glob.glob(
        'C:\\Repos\\Udacity\\project1\\data\\processed\\*.tfrecord')
    sample_path = [all_tf_records[0]]
    # "C:\\Repos\\Udacity\\project1\\data\\processed\\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

    # read_plot_tf_records(sample_path)
    from_example(all_tf_records[0:4])
    # input("Press Enter to continue...")

    # run_dataset_example('./data/small_images.tfrecords')
