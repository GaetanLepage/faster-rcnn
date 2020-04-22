import tensorflow as tf

def get_base_model():

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
                                                   include_top=False)

    # Feature map layer
    output_layer = base_model.get_layer('block_6_expand_relu')

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input,
                                outputs=output_layer.output,
                                name='feature_extractor')

    down_stack.trainable = False

    # down_stack.summary()

    # tf.keras.utils.plot_model(down_stack,
                              # to_file="model.png",
                              # show_shapes=True)

    return down_stack


def rpn_model(sliding_window_size, num_anchors):
    """
    RPN model builder

    TODO remove : Builds the global Faster-RCNN model

    Args:
        sliding_window_size:    the size of the sliding window (n in the paper)
        num_windows:            the number of anchors

    Returns:
        a tf.keras.Model including the feature extractor and the object detection patches
    """

    inputs = tf.keras.Input(shape=(None, None, 3), name='input_layer')

    feature_extractor = get_base_model()

    feature_map = feature_extractor(inputs)

    # n_channels = feature_extractor.output_shape[-1]
    n_channels = feature_map.shape[-1]

    sliding_windows_features = tf.keras.layers.Conv2D(filters=n_channels,
                                                      kernel_size=sliding_window_size,
                                                      name='rpn_intermediate_layer')(feature_map)

    cls_output = tf.keras.layers.Conv2D(filters=2 * num_anchors,
                                        kernel_size=1,
                                        name='rpn_cls')(sliding_windows_features)

    # Softmax layer to deal with class probabilities
    cls_output = tf.keras.layers.Softmax()

    reg_output = tf.keras.layers.Conv2D(filters=4 * num_anchors,
                                        kernel_size=1,
                                        name='rpn_reg')(sliding_windows_features)

    return tf.keras.Model(inputs=inputs,
                          outputs=[cls_output, reg_output])
