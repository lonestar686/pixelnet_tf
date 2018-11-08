import tensorflow as tf
import tensorflow.contrib.slim as slim


def random_sampling(features, labels, index):
    with tf.name_scope('RandomSampling'):
        shape = features[0].shape[1:-1]
        upsampled = [features[0]]
        for i in range(1, len(features)):
            upsampled.append(tf.image.resize_bilinear(features[i], shape))
        if index is None:
            vector = tf.concat(upsampled, axis=-1)
            label = None
        else:
            sampled = [tf.gather_nd(feature, index) for feature in upsampled]
            vector = tf.concat(sampled, axis=-1)
            label = tf.gather_nd(labels, index)
        return vector, label


def pixelnet(images, labels, index=None, num_classes=21):
    with tf.name_scope('PixelNet'):
        features = []
        x = images
        x = slim.conv2d(x, 64, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv1_1')
        x = slim.conv2d(x, 64, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv1_2')
        features.append(x)
        x = slim.max_pool2d(x, [2, 2], scope='pool1')
        x = slim.conv2d(x, 128, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv2_1')
        x = slim.conv2d(x, 128, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv2_2')
        features.append(x)
        x = slim.max_pool2d(x, [2, 2], scope='pool2')
        x = slim.conv2d(x, 256, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv3_1')
        x = slim.conv2d(x, 256, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv3_2')
        x = slim.conv2d(x, 256, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv3_3')
        features.append(x)
        x = slim.max_pool2d(x, [2, 2], scope='pool3')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv4_1')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv4_2')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv4_3')
        features.append(x)
        x = slim.max_pool2d(x, [2, 2], scope='pool4')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv5_1')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv5_2')
        x = slim.conv2d(x, 512, [3, 3], padding='SAME', activation_fn=tf.nn.relu, scope='conv5_3')
        features.append(x)
        x = slim.max_pool2d(x, [2, 2], scope='pool5')
        x = slim.conv2d(x, 4096, [7, 7], padding='VALID', activation_fn=tf.nn.relu, scope='fc6')
        x = slim.dropout(x, 0.5, scope='dropout6')
        x = slim.conv2d(x, 4096, [1, 1], padding='VALID', activation_fn=tf.nn.relu, scope='fc7')
        x = slim.dropout(x, 0.5, scope='dropout7')
        features.append(x)
        x, y = random_sampling(features, labels, index)
        with tf.name_scope('MLP'):
            x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu, scope='fc1')
            x = slim.dropout(x, 0.5, scope='dropout1')
            x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu, scope='fc2')
            x = slim.dropout(x, 0.5, scope='dropout2')
            x = slim.fully_connected(x, num_classes, activation_fn=tf.nn.relu, scope='fc3')
        if labels is not None:
            return x, y
        else:
            return x
