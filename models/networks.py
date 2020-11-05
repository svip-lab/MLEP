import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np

from models import pix2pix


def cyclegan_arg_scope(instance_norm_center=True,
                       instance_norm_scale=True,
                       instance_norm_epsilon=0.001,
                       weights_init_stddev=0.02,
                       weight_decay=0.0):
    """Returns a default argument scope for all generators and discriminators.
    Args:
      instance_norm_center: Whether instance normalization applies centering.
      instance_norm_scale: Whether instance normalization applies scaling.
      instance_norm_epsilon: Small float added to the variance in the instance
        normalization to avoid dividing by zero.
      weights_init_stddev: Standard deviation of the random values to initialize
        the convolution kernels with.
      weight_decay: Magnitude of weight decay applied to all convolution kernel
        variables of the generator.
    Returns:
      An arg-scope.
    """
    instance_norm_params = {
        'center': instance_norm_center,
        'scale': instance_norm_scale,
        'epsilon': instance_norm_epsilon,
    }

    weights_regularizer = None
    if weight_decay and weight_decay > 0.0:
        weights_regularizer = tf_layers.l2_regularizer(weight_decay)

    with tf.contrib.framework.arg_scope(
            [tf_layers.conv2d, tf_layers.conv3d],
            normalizer_fn=tf_layers.instance_norm,
            normalizer_params=instance_norm_params,
            weights_initializer=tf.random_normal_initializer(0, weights_init_stddev),
            weights_regularizer=weights_regularizer) as sc:
        return sc


def cyclegan_upsample(net, num_outputs, stride, method='conv2d_transpose'):
    """Upsamples the given inputs.
    Args:
      net: A Tensor of size [batch_size, height, width, filters].
      num_outputs: The number of output filters.
      stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
        relative to the inputs, of the output dimensions. For example, if kernel
        size is [2, 3], then the output height and width will be twice and three
        times the input size.
      method: The upsampling method: 'nn_upsample_conv', 'bilinear_upsample_conv',
        or 'conv2d_transpose'.
    Returns:
      A Tensor which was upsampled using the specified method.
    Raises:
      ValueError: if `method` is not recognized.
    """
    with tf.variable_scope('upconv'):
        net_shape = tf.shape(net)
        height = net_shape[1]
        width = net_shape[2]

        # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a 3x3
        # 'valid' convolution produce an output with the same dimension as the
        # input.
        spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

        if method == 'nn_upsample_conv':
            net = tf.image.resize_nearest_neighbor(
                net, [stride[0] * height, stride[1] * width])
            net = tf.pad(net, spatial_pad_1, 'REFLECT')
            net = tf_layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        if method == 'bilinear_upsample_conv':
            net = tf.image.resize_bilinear(
                net, [stride[0] * height, stride[1] * width])
            net = tf.pad(net, spatial_pad_1, 'REFLECT')
            net = tf_layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        elif method == 'conv2d_transpose':
            net = tf_layers.conv2d_transpose(
                net, num_outputs, kernel_size=[3, 3], stride=stride, padding='same')
        else:
            raise ValueError('Unknown method: [%s]', method)

        return net


def tensor_split_times(tensor, batch_size, time_steps):
    """
    :param tensor: (N x T) x h x w x c
    :return: N x T x h x w x c
    """
    return tf.reshape(tensor, shape=[batch_size, time_steps] + tensor.get_shape().as_list()[1:])


def tensor_fuse_times(tensor, batch_size, time_steps):
    """
    :param tensor: N x T x h x w x c
    :return: (N x T) x h x w x c
    """
    return tf.reshape(tensor, shape=[batch_size * time_steps] + tensor.get_shape().as_list()[2:])


def tensor_stack_times(tensor, time_steps):
    split_times = []
    for i in range(time_steps):
        split_times.append(tensor[:, i, ...])

    return tf.concat(split_times, axis=3)


def recurrent_model(net_split_times, spatial_shape, batch_size, num_outputs):

    conv_lstm_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=spatial_shape,
                                                 output_channels=num_outputs, kernel_shape=[3, 3])
    initial_state = conv_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    final_outputs, final_states = tf.nn.dynamic_rnn(cell=conv_lstm_cell, inputs=net_split_times,
                                                    initial_state=initial_state, time_major=False,
                                                    scope='ConvLSTM')
    # take the T time step output
    hidden_state = final_states[-1]

    return hidden_state


def resnet_convlstm(inputs, num_filters=64,
                    upsample_fn=cyclegan_upsample,
                    kernel_size=3,
                    num_outputs=3,
                    tanh_linear_slope=0.0,
                    use_decoder=False):

    batch_size, time_steps, height, width, channel = inputs.get_shape().as_list()

    inputs = tensor_fuse_times(inputs, batch_size, time_steps)

    end_points = {}

    if height and height % 4 != 0:
        raise ValueError('The input height must be a multiple of 4.')
    if width and width % 4 != 0:
        raise ValueError('The input width must be a multiple of 4.')

    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size, kernel_size]

    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height // 2
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width // 2
    paddings = np.array(
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        dtype=np.int32)
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):

        ###########
        # Encoder #
        ###########
        with tf.variable_scope('encoder'):
            # 7x7 input stage, 224 x 224 x 64
            net = tf.pad(inputs, spatial_pad_3, 'REFLECT')
            net = tf_layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
            end_points['resnet_stage_0'] = net

            # 3x3 state, 112 x 112 x 128
            net = tf.pad(net, paddings, 'REFLECT')
            net = tf_layers.conv2d(net, num_filters * 2, kernel_size=kernel_size, stride=2,
                                   activation_fn=tf.nn.relu, padding='VALID')
            end_points['resnet_stage_1'] = net

            #########################################
            # 3 Residual Blocks with  56 x 56 x 256 #
            #########################################
            with tf.variable_scope('residual_blocks'):
                with tf.contrib.framework.arg_scope(
                        [tf_layers.conv2d],
                        kernel_size=kernel_size,
                        stride=1,
                        activation_fn=tf.nn.relu,
                        padding='VALID'):
                    net = tf.pad(net, paddings, 'REFLECT')
                    net = tf_layers.conv2d(net, num_filters * 4, kernel_size=kernel_size, stride=2,
                                           activation_fn=tf.nn.relu, padding='VALID')
                    end_points['resnet_stage_2'] = net
                    for block_id in range(3):
                        with tf.variable_scope('stage_2_block_{}'.format(block_id)):
                            res_net = tf.pad(net, paddings, 'REFLECT')
                            res_net = tf_layers.conv2d(res_net, num_filters * 4)
                            res_net = tf.pad(res_net, paddings, 'REFLECT')
                            res_net = tf_layers.conv2d(res_net, num_filters * 4,
                                                       activation_fn=None)
                            net += res_net

                            end_points['resnet_state_2_block_%d' % block_id] = net

                    #########################################
                    # 4 Residual Blocks with  28 x 28 x 512 #
                    #########################################
                    net = tf.pad(net, paddings, 'REFLECT')
                    net = tf_layers.conv2d(net, num_filters * 8, kernel_size=kernel_size, stride=2,
                                           activation_fn=tf.nn.relu, padding='VALID')
                    end_points['resnet_stage_3'] = net
                    for block_id in range(3):
                        with tf.variable_scope('stage_3_block_{}'.format(block_id)):
                            res_net = tf.pad(net, paddings, 'REFLECT')
                            res_net = tf_layers.conv2d(res_net, num_filters * 8)
                            res_net = tf.pad(res_net, paddings, 'REFLECT')
                            res_net = tf_layers.conv2d(res_net, num_filters * 8,
                                                       activation_fn=None)
                            net += res_net

                            end_points['resnet_state_3_block_%d' % block_id] = net

        ####################
        # Recurrent module #
        ####################
        with tf.variable_scope('recurrent'):
            # reshape net to N x T x h x w x c
            spatial_shape = net.get_shape().as_list()[1:]
            net_split_times = tensor_split_times(net, batch_size=batch_size, time_steps=time_steps)
            print('Encoder output = {}'.format(net_split_times))

            hidden_state = recurrent_model(net_split_times, spatial_shape, batch_size, num_filters * 8)
            end_points['hidden_state'] = hidden_state
            print('ConvLSTM hidden state = {}', hidden_state)

        ###########
        # Decoder #
        ###########
        with tf.variable_scope('decoder'):

            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu):

                with tf.variable_scope('decoder1'):
                    net = upsample_fn(hidden_state, num_outputs=num_filters * 8, stride=[2, 2])
                end_points['decoder1'] = net

                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters * 4, stride=[2, 2])
                end_points['decoder2'] = net

                with tf.variable_scope('decoder3'):
                    net = upsample_fn(net, num_outputs=num_filters * 2, stride=[2, 2])
                end_points['decoder3'] = net

        with tf.variable_scope('output'):
            net = tf.pad(net, spatial_pad_3, 'REFLECT')
            logits = tf_layers.conv2d(
                net,
                num_outputs, [7, 7],
                activation_fn=None,
                normalizer_fn=None,
                padding='valid')
            # logits = tf.reshape(logits, _dynamic_or_static_shape(images))

            end_points['logits'] = logits
            end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope
    print('Decoder output = {}'.format(logits))

    return end_points['predictions'], end_points['hidden_state'], end_points


def deconv_module(hidden_state, num_filters=64, num_outputs=3, kernel_size=3,
                  upsample_fn=cyclegan_upsample, tanh_linear_slope=0.0):

    end_points = {}
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])
    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):
        with tf.variable_scope('decoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu):
                with tf.variable_scope('decoder1'):
                    net = upsample_fn(hidden_state, num_outputs=num_filters * 2, stride=[2, 2])
                    end_points['decoder1'] = net

                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
                    end_points['decoder2'] = net

        with tf.variable_scope('output'):
            net = tf.pad(net, spatial_pad_3, 'REFLECT')
            logits = tf_layers.conv2d(
                net,
                num_outputs, [7, 7],
                activation_fn=None,
                normalizer_fn=None,
                padding='valid')

            outputs = tf.tanh(logits) + logits * tanh_linear_slope
        # print('Decoder output = {}'.format(logits))

    return outputs, end_points


def cyclegan_convlstm(inputs, num_filters=64,
                      upsample_fn=cyclegan_upsample,
                      kernel_size=3,
                      num_outputs=3,
                      tanh_linear_slope=0.0, use_decoder=False):

    batch_size, time_steps, height, width, channel = inputs.get_shape().as_list()

    inputs = tensor_fuse_times(inputs, batch_size, time_steps)

    if height and height % 4 != 0:
        raise ValueError('The input height must be a multiple of 4.')
    if width and width % 4 != 0:
        raise ValueError('The input width must be a multiple of 4.')

    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size, kernel_size]

    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height // 2
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width // 2
    paddings = np.array(
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        dtype=np.int32)
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):

        ###########
        # Encoder #
        ###########
        with tf.variable_scope('input'):
            # 7x7 input stage
            net = tf.pad(inputs, spatial_pad_3, 'REFLECT')
            net = tf_layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')

        with tf.variable_scope('encoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=2,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                net = tf.pad(net, paddings, 'REFLECT')
                net = tf_layers.conv2d(net, num_filters * 2)

                net = tf.pad(net, paddings, 'REFLECT')
                net = tf_layers.conv2d(net, num_filters * 4)

        ###################
        # Residual Blocks #
        ###################
        with tf.variable_scope('residual_blocks'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                for block_id in range(6):
                    with tf.variable_scope('block_{}'.format(block_id)):
                        res_net = tf.pad(net, paddings, 'REFLECT')
                        res_net = tf_layers.conv2d(res_net, num_filters * 4)
                        res_net = tf.pad(res_net, paddings, 'REFLECT')
                        res_net = tf_layers.conv2d(res_net, num_filters * 4, activation_fn=None)
                        net += res_net

        ####################
        # Recurrent module #
        ####################
        with tf.variable_scope('recurrent'):
            # reshape net to N x T x h x w x c
            spatial_shape = net.get_shape().as_list()[1:]
            net_split_times = tensor_split_times(net, batch_size=batch_size, time_steps=time_steps)
            # print('Encoder output = {}'.format(net_split_times))

            hidden_state = recurrent_model(net_split_times, spatial_shape, batch_size, num_filters * 4)
            # print('ConvLSTM hidden state = {}', hidden_state)

        ###########
        # Decoder #
        ###########
        if use_decoder:
            outputs, end_points = deconv_module(hidden_state, num_filters, num_outputs=num_outputs,
                                                upsample_fn=upsample_fn, tanh_linear_slope=tanh_linear_slope)
        else:
            outputs, end_points = None, {}
        end_points['hidden_state'] = hidden_state

    return outputs, hidden_state, end_points


def cyclegan_convlstm_deconv1(inputs, num_filters=64,
                              upsample_fn=cyclegan_upsample,
                              kernel_size=3,
                              num_outputs=3,
                              tanh_linear_slope=0.0, use_decoder=False):

    _, hidden_state, end_points = cyclegan_convlstm(inputs, use_decoder=False)

    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):
        with tf.contrib.framework.arg_scope(
                [tf_layers.conv2d_transpose],
                kernel_size=kernel_size,
                stride=4,
                padding='SAME'):
            with tf.variable_scope('decoder1'):
                net = tf_layers.conv2d_transpose(
                    hidden_state,
                    num_filters,
                    normalizer_fn=None)

            with tf.variable_scope('output'):
                logits = tf_layers.conv2d(
                    net,
                    num_outputs, [7, 7],
                    activation_fn=None,
                    normalizer_fn=None,
                    padding='SAME')

            outputs = tf.tanh(logits) + logits * tanh_linear_slope

    return outputs, hidden_state, end_points


def two_cyclegan_convlstm_classifier(inputs, is_training=True, keep_prob=0.8, weight_decay=0.004):
    _, hidden_state, _ = cyclegan_convlstm(inputs=inputs)

    with tf.contrib.framework.arg_scope([tf_layers.fully_connected],
                                        weights_regularizer=tf_layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope([tf_layers.dropout], is_training=is_training, keep_prob=keep_prob):
            with tf.variable_scope('fc'):
                net = tf_layers.flatten(hidden_state)
                net = tf_layers.dropout(net)
                net = tf_layers.fully_connected(inputs=net, num_outputs=128, activation_fn=tf.nn.relu)

        with tf.variable_scope('logits'):
            logits = tf_layers.fully_connected(inputs=net, num_outputs=2, activation_fn=None)
            probabilities = tf.nn.softmax(logits, name='prob')

    return logits, probabilities


def cyclegan_conv2d(inputs, num_filters=64,
                    upsample_fn=cyclegan_upsample,
                    kernel_size=3, num_outputs=3, tanh_linear_slope=0.0, use_decoder=True):

    batch_size, time_steps, height, width, channel = inputs.get_shape().as_list()

    inputs = tensor_stack_times(inputs, time_steps)

    end_points = {}

    if height and height % 4 != 0:
        raise ValueError('The input height must be a multiple of 4.')
    if width and width % 4 != 0:
        raise ValueError('The input width must be a multiple of 4.')

    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size, kernel_size]

    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height // 2
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width // 2
    paddings = np.array(
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        dtype=np.int32)
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):

        ###########
        # Encoder #
        ###########
        with tf.variable_scope('input'):
            # 7x7 input stage
            net = tf.pad(inputs, spatial_pad_3, 'REFLECT')
            net = tf_layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
            end_points['encoder_0'] = net

        with tf.variable_scope('encoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=2,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                net = tf.pad(net, paddings, 'REFLECT')
                net = tf_layers.conv2d(net, num_filters * 2)
                end_points['encoder_1'] = net
                net = tf.pad(net, paddings, 'REFLECT')
                net = tf_layers.conv2d(net, num_filters * 4)
                end_points['encoder_2'] = net

        ###################
        # Residual Blocks #
        ###################
        with tf.variable_scope('residual_blocks'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                for block_id in range(6):
                    with tf.variable_scope('block_{}'.format(block_id)):
                        res_net = tf.pad(net, paddings, 'REFLECT')
                        res_net = tf_layers.conv2d(res_net, num_filters * 4)
                        res_net = tf.pad(res_net, paddings, 'REFLECT')
                        res_net = tf_layers.conv2d(res_net, num_filters * 4,
                                                   activation_fn=None)
                        net += res_net

                        end_points['resnet_block_%d' % block_id] = net

        end_points['hidden_state'] = net

        ###########
        # Decoder #
        ###########
        with tf.variable_scope('decoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu):
                with tf.variable_scope('decoder1'):
                    net = upsample_fn(net, num_outputs=num_filters * 2, stride=[2, 2])
                end_points['decoder1'] = net

                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
                end_points['decoder2'] = net

        with tf.variable_scope('output'):
            net = tf.pad(net, spatial_pad_3, 'REFLECT')
            logits = tf_layers.conv2d(
                net,
                num_outputs, [7, 7],
                activation_fn=None,
                normalizer_fn=None,
                padding='valid')

            end_points['logits'] = logits
            end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope
        print('Decoder output = {}'.format(logits))

    return end_points['predictions'], end_points['hidden_state'], end_points


def unet_conv2d(inputs, num_filters=64, num_down_samples=4, use_decoder=True):
    _, time_steps, _, _, _ = inputs.get_shape().as_list()

    in_node = tensor_stack_times(inputs, time_steps)
    conv = []
    for layer in range(0, num_down_samples):
        features = 2**layer*num_filters

        conv1 = tf_layers.conv2d(inputs=in_node, num_outputs=features, kernel_size=3)
        # if layer == num_down_samples - 1:
        #     conv2 = tf_layers.conv2d(inputs=conv1, num_outputs=features, kernel_size=3, activation_fn=tf.nn.tanh)
        # else:
        #     conv2 = tf_layers.conv2d(inputs=conv1, num_outputs=features, kernel_size=3, activation_fn=tf.nn.relu)
        conv2 = tf_layers.conv2d(inputs=conv1, num_outputs=features, kernel_size=3, activation_fn=tf.nn.relu)

        conv.append(conv2)

        if layer < num_down_samples - 1:
            in_node = tf_layers.max_pool2d(inputs=conv2, kernel_size=2, padding='SAME')
            # in_node = conv2d(inputs=conv2, num_outputs=features, kernel_size=filter_size, stride=2)

    in_node = conv[-1]
    hidden_state = conv[-1]

    for layer in range(num_down_samples-2, -1, -1):
        features = 2**(layer+1)*num_filters

        h_deconv = tf_layers.conv2d_transpose(inputs=in_node, num_outputs=features//2, kernel_size=2, stride=2)
        h_deconv_concat = tf.concat([conv[layer], h_deconv], axis=3)

        conv1 = tf_layers.conv2d(inputs=h_deconv_concat, num_outputs=features//2, kernel_size=3)
        in_node = tf_layers.conv2d(inputs=conv1, num_outputs=features//2, kernel_size=3)

    output = tf_layers.conv2d(inputs=in_node, num_outputs=3, kernel_size=3, activation_fn=None)
    output = tf.tanh(output)
    return output, hidden_state, None


def unet_conv2d_instance_norm(inputs, num_filters=64, num_down_samples=4, use_decoder=True):
    _, time_steps, _, _, _ = inputs.get_shape().as_list()

    in_node = tensor_stack_times(inputs, time_steps)
    conv = []
    end_points = {}
    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):
        for layer in range(0, num_down_samples):
            features = 2**layer*num_filters

            conv1 = tf_layers.conv2d(inputs=in_node, num_outputs=features, kernel_size=3)
            conv2 = tf_layers.conv2d(inputs=conv1, num_outputs=features, kernel_size=3)
            conv.append(conv2)

            if layer < num_down_samples - 1:
                in_node = tf_layers.max_pool2d(inputs=conv2, kernel_size=2, padding='SAME')
                # in_node = conv2d(inputs=conv2, num_outputs=features, kernel_size=filter_size, stride=2)

        in_node = conv[-1]
        hidden_state = conv[-1]

        if use_decoder:
            for i, layer in enumerate(range(num_down_samples-2, -1, -1)):
                features = 2**(layer+1)*num_filters

                h_deconv = tf_layers.conv2d_transpose(inputs=in_node, num_outputs=features//2, kernel_size=2, stride=2)
                h_deconv_concat = tf.concat([conv[layer], h_deconv], axis=3)

                conv1 = tf_layers.conv2d(inputs=h_deconv_concat, num_outputs=features//2, kernel_size=3)
                in_node = tf_layers.conv2d(inputs=conv1, num_outputs=features//2, kernel_size=3)

                end_points['encoder_%d' % i] = in_node

            output = tf_layers.conv2d(inputs=in_node, num_outputs=3, kernel_size=3, activation_fn=None)
            output = tf.tanh(output)
        else:
            output = None
    return output, hidden_state, end_points


def conv2d_deconv2d(inputs, num_filters=64, num_down_samples=4):
    _, time_steps, _, _, _ = inputs.get_shape().as_list()

    in_node = tensor_stack_times(inputs, time_steps)
    for layer in range(0, num_down_samples):
        features = 2**layer*num_filters

        conv1 = tf_layers.conv2d(inputs=in_node, num_outputs=features, kernel_size=3)
        conv2 = tf_layers.conv2d(inputs=conv1, num_outputs=features, kernel_size=3)

        if layer < num_down_samples - 1:
            in_node = tf_layers.max_pool2d(inputs=conv2, kernel_size=2, padding='SAME')
            # in_node = conv2d(inputs=conv2, num_outputs=features, kernel_size=filter_size, stride=2)

    in_node = conv2
    hidden_state = conv2

    for layer in range(num_down_samples-2, -1, -1):
        features = 2**(layer+1)*num_filters

        h_deconv = tf_layers.conv2d_transpose(inputs=in_node, num_outputs=features//2, kernel_size=2, stride=2)

        conv1 = tf_layers.conv2d(inputs=h_deconv, num_outputs=features//2, kernel_size=3)
        in_node = tf_layers.conv2d(inputs=conv1, num_outputs=features//2, kernel_size=3)

    output = tf_layers.conv2d(inputs=in_node, num_outputs=3, kernel_size=3, activation_fn=None)
    output = tf.tanh(output)
    return output, hidden_state


def resnet_conv3d(inputs, num_filters=64,
                  upsample_fn=cyclegan_upsample,
                  kernel_size=3,
                  num_outputs=3,
                  tanh_linear_slope=0.0, use_decoder=True):

    end_points = {}

    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):

        ###########
        # Encoder #
        ###########
        with tf.variable_scope('input'):
            # 7x7 input stage
            net = tf_layers.conv3d(inputs, num_filters, kernel_size=[1, 7, 7], stride=1, padding='SAME')
            end_points['encoder_0'] = net

        with tf.variable_scope('encoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv3d],
                    kernel_size=kernel_size,
                    stride=2,
                    activation_fn=tf.nn.relu,
                    padding='SAME'):
                net = tf_layers.conv3d(net, num_filters * 2)
                end_points['encoder_1'] = net
                net = tf_layers.conv3d(net, num_filters * 4)
                end_points['encoder_2'] = net

        ###################
        # Residual Blocks #
        ###################
        with tf.variable_scope('residual_blocks'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv3d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    padding='SAME'):
                for block_id in range(6):
                    with tf.variable_scope('block_{}'.format(block_id)):
                        res_net = tf_layers.conv3d(net, num_filters * 4)
                        res_net = tf_layers.conv3d(res_net, num_filters * 4,
                                                   activation_fn=None)
                        net += res_net

                        end_points['resnet_block_%d' % block_id] = net

        hidden_state = tf.nn.tanh(net)
        hidden_state = tf.squeeze(hidden_state, axis=1)
        end_points['hidden_state'] = hidden_state

        ###########
        # Decoder #
        ###########
        with tf.variable_scope('decoder'):
            with tf.contrib.framework.arg_scope(
                    [tf_layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu):
                with tf.variable_scope('decoder1'):
                    net = upsample_fn(hidden_state, num_outputs=num_filters * 2, stride=[2, 2])
                end_points['decoder1'] = net

                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
                end_points['decoder2'] = net

        with tf.variable_scope('output'):
            logits = tf_layers.conv2d(
                net,
                num_outputs, [7, 7],
                activation_fn=None,
                normalizer_fn=None,
                padding='SAME')

            end_points['logits'] = logits
            end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope
        print('Decoder output = {}'.format(logits))

    return end_points['predictions'], end_points['hidden_state'], end_points


def discriminator(inputs, num_filers=(128, 256, 512, 512)):
    logits, end_points = pix2pix.pix2pix_discriminator(inputs, num_filers)
    return logits, end_points['predictions']


if __name__ == '__main__':
    input_tensor = tf.placeholder(shape=[10, 8, 224, 224, 3], dtype=tf.float32)
    # logits, end_points = cyclegan_convlstm(inputs=input_tensor, num_outputs=3, num_filters=64)
    resnet_conv3d(inputs=input_tensor, num_outputs=3, num_filters=64)
