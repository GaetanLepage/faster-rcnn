"""
TODO
"""

import sys
import logging
sys.path.append(".")

import tensorflow as tf
import tensorflow_datasets as tfds

import faster_rcnn.data as data
from faster_rcnn.rpn.rpn_model import get_rpn_model
from faster_rcnn.rpn.rpn_loss import rpn_loss_wrapper


RPN_SLIDING_WINDOW_SIZE = 3
# RPN


def train_faster_rcnn(rpn_sliding_window_size,
                      anchors_area_list,
                      anchors_aspect_ratio_list):
    """
    Perform the 4-step alternating training described in the Faster-RCNN paper.
        * Step 1: RPN training (train_rpn)
        * Step 2: Fast-RCNN training (train_fast_rcnn)
        * Step 3: Fine tune RPN with the common convolutional layers
        * Step 4: Fine tune Fast-RCNN with the common convolutional layers

    Args:
        rpn_sliding_window_size: The size of the sliding window of the RPN (n in the paper).
        anchors_area_list: The list of the different areas of the anchors (RPN network).
        anchors_aspect_ratio_list: The lis tof the aspect ratios of the anchors (RPN network).
    """

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # instanciate RPN model
        rpn_model = get_rpn_model(sliding_window_size=rpn_sliding_window_size,
                                  num_anchors=len(anchors_area_list) * len(anchors_aspect_ratio_list))

        # instanciate the loss
        rpn_loss = rpn_loss_wrapper(lambda_balance=10)
        # compile the model
        rpn_model.compile(optimizer='adam',
                          loss=rpn_loss)

    train_rpn(rpn_model=rpn_model,
              batch_size=1,
              feature_map_shape=(16, 16),
              anchors_area_list=anchors_area_list,
              anchors_aspect_ratio_list=anchors_aspect_ratio_list)


def train_rpn(rpn_model,
              batch_size,
              feature_map_shape,
              anchors_area_list,
              anchors_aspect_ratio_list):
    """
    """

    train_dataset, info = tfds.load(name='voc',
                                    split='train',
                                    shuffle_files=True,
                                    with_info=True)


    train_dataset = train_dataset.shuffle(info.splits['train'].num_examples)

    logging.info("RPN training : Pre processing training data")

    # train_dataset = train_dataset.map(lambda datapoint: (datapoint['image'], datapoint['objects']['bbox']))

    # load_function = data.get_load_image_function(feature_map_shape=feature_map_shape,
                                                 # anchors_area_list=anchors_area_list,
                                                 # anchors_aspect_ratio_list=anchors_aspect_ratio_list)

    # train_dataset = train_dataset.map(lambda *datapoint: tf.py_function(load_function,
                                                                        # inp=datapoint,
                                                                        # Tout=(tf.bool, tf.bool, tf.float32)),
                                      # num_parallel_calls=tf.data.experimental.AUTOTUNE)


    train_dataset = train_dataset.map(lambda datapoint: tf.py_function(data.load_image_train,
                                                                       inp=[datapoint['image'],
                                                                            datapoint['objects']['bbox'],
                                                                            feature_map_shape,
                                                                            anchors_area_list,
                                                                            anchors_aspect_ratio_list],
                                                                       Tout=(tf.bool, tf.bool, tf.float32)),
                                      num_parallel_calls=1)
                                      # num_parallel_calls=tf.data.experimental.AUTOTUNE)


    train_dataset = train_dataset.batch(batch_size)

    for x in train_dataset.take(1):
        print(x)

    rpn_model.fit(x=data)


def train_fast_rcnn():
    """
    """

def fine_tune_rpn():
    """
    """

def fine_tune_fast_rcnn():
    """
    """


# TODO remove
if __name__ == "__main__":
    train_faster_rcnn(rpn_sliding_window_size=3,
                      anchors_area_list=[128**2, 256**2, 512**2],
                      anchors_aspect_ratio_list=[0.5, 1, 2])
