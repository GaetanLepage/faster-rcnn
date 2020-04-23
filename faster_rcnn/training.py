import tensorflow as tf

from faster_rcnn.rpn.rpn_model import get_rpn_model
from faster_rcnn.rpn.rpn_loss import rpn_loss_wrapper


RPN_SLIDING_WINDOW_SIZE = 3
# RPN


def train_faster_rcnn(rpn_sliding_window_size,
                      rpn_num_anchors):
    """
    """

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # instanciate RPN model
        rpn_model = get_rpn_model(sliding_window_size=rpn_sliding_window_size,
                                  num_anchors=rpn_num_anchors)

        # instanciate the loss
        rpn_loss = rpn_loss_wrapper(lambda_balance=10)
        # compile the model
        rpn_model.compile(optimizer='adam',
                          loss=rpn_loss)

    train_rpn(rpn_model, data)


def train_rpn(rpn_model, data):
    """
    """

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
