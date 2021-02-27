"""
Custom Data Loader for Cassava Leaf Classification Project
"""

__author__ = "Maitreya Venkataswamy"

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def initialize_dataset(tfrecord_files, batch_size=1, labelled=True,
                       img_size=None, flip=False, rot=(0, 0)):
    """Creates a Tensorflow Dataset from a list of TFRecord filenames"""
    # Create the initial dataset with the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)

    # Define the parsing function for each example
    @tf.autograph.experimental.do_not_convert
    def _parse_img_fn(example):
        """Parsing function for a single example"""
        # Define datatypes of each element of the example
        img_feat_desc = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string)
        }

        # If the data is labelled, add a description for the labelled data
        if labelled:
            img_feat_desc["target"] = tf.io.FixedLenFeature([], tf.int64)

        # Parse the example with the parsing function
        example = tf.io.parse_single_example(example, img_feat_desc)

        # Extract the image data from the parsed example
        img = example["image"]

        # Decode the image into a Tensor
        img = tf.image.decode_jpeg(img, channels=3)

        # Cast the image tensor into Float32
        img = tf.cast(img, tf.float32)

        # Extract the label from the example and into Int32
        if labelled:
            label = example["target"]
            label = tf.cast(label, tf.int32)

        # Return the image and the label
        if labelled:
            return img, label
        else:
            return img

    # Apply the parsing function to every element in the dataset
    dataset = raw_dataset.map(
        _parse_img_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Prefetch the data
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Apply a batch size to the dataset
    dataset = dataset.batch(batch_size)

    # Initialize the data augmentation pipeline
    data_aug = tf.keras.Sequential()

    # Add resizing if specified
    if img_size is not None:
        data_aug.add(preprocessing.Resizing(*img_size))

    # Add random flipping if specified
    if flip:
        data_aug.add(preprocessing.RandomFlip())

    # Add random rotations
    data_aug.add(preprocessing.RandomRotation(rot))

    # Apply the data augmentation pipeline
    if labelled:
        dataset = dataset.map(
            lambda img, label: (data_aug(img), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda img: data_aug(img),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Return the dataset
    return dataset
