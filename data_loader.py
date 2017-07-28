"""
implemented over
- https://github.com/carpedm20/BEGAN-tensorflow/blob/master/data_loader.py
"""

import os
from PIL import Image
from glob import glob
import tensorflow as tf

from data_reader import *

def get_loader(root, batch_size, low_scale = 64, high_scale = 256, data_format, split=None, is_grayscale=False, seed=None):
    # Check that dataset type is valid
    dataset = dataset.lower()
    assert dataset == 'cub' or dataset == 'oxford-102', \
        "Dataset not recognized: %s. Must be 'cub' or 'oxford-102'" % dataset

    self.dataset = dataset

    # Initialize path, if not specified
    if path is None:
        self.path =  './datasets/' + self.__dataset

    # Check if path exists
    assert os.path.exists(self.path), "Path %s does not exist" % self.path

    # Check images_and_texts folder
    data_path = os.path.join(self.path + "/images_and_texts/")
    assert data_path, "Didn't find 'images_and_texts' folder in %s" % self.path

    # images_and_texts folder contains subfolders
    folder_list = os.listdir(data_path)
    folder_list.sort()

    # file path list
    img_paths = []
    txt_paths = []

    for folder in folder_list:
        img_paths.append(glob("{}/*.{}".format(data_path, '.jpg')))
        txt_paths.append(glob("{}/*.{}".format(data_path, '.txt')))


    assert len(img_paths) == 0 or len(txt_paths) == 0 , 'No image/text file found in paths'

    # set jpg decoder
    tf_decode = tf.image.decode_jpeg


    with Image.open(img_paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]


    img_filename_queue, text_filename_queue = tf.train.string_input_producer([img_paths, txt_paths], shuffle=False, seed=seed)
    # img file
    img_reader = tf.WholeFileReader()
    filename, data = img_reader.read(img_filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    # text file
    txt_reader = tf.TextLineReader()
    txt_filename, txt_data = txt_reader.read(text_filename_queue)
    text = tf.decode_raw(txt_data, tf.float32)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    img_queue, txt_queue = tf.train.shuffle_batch(
        [image, text], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    img_queue = tf.image.resize_nearest_neighbor(img_queue, [low_scale, low_scale])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(img_queue), tf.to_float(txt_queue)
