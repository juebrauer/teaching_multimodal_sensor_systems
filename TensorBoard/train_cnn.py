# file: train_cnn.py
#
# A CNN implementation in TensorFlow
#
# First set the variable <dataset_root>
# It tells the Python script where your training and
# validation data is.
#
# Store your training images in a sub-folder for each
# class in <dataset_root>/train/
# e.g.
#    <dataset_root>/train/bikes
#    <dataset_root>/train/cars
#
# Store your test images in a sub-folder for each
# class in <dataset_root>/test/
# e.g.
#    <dataset_root>/test/bikes
#    <dataset_root>/test/cars
#
# Then let the script run.
# The final model will be saved.
#
# ---
# Prof. Dr. Juergen Brauer, www.juergenbrauer.org

from dataset_reader import dataset
import tensorflow as tf
import numpy as np
import cv2

# Experiments
# Exp-Nr   Comment
# 01       10 hidden layers: INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
#          trained 100 mini-batches of size 32
#          feature maps: 10-15-20-25-30
#          --> 59.93% accuracy evaluated with test_cnn.py
#
# 02       10 hidden layers: INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
#          trained 2000 mini-batches of size 32
#          feature maps: 10-15-20-25-30
#          --> 72.50% accuracy evaluated with test_cnn.py
#
# 03       4 hidden layers: INPUT->C1->P1->FC1->FC2->OUT
#          trained 1000 mini-batches of size 32
#          feature maps: 10
#          --> 66.51% accuracy evaluated with test_cnn.py (using gray-scale images)
#
#

exp_nr = 1
n_classes = 2
imagesize = 227
nr_img_channels = 3
learn_rate = 1
batch_size = 32
if exp_nr==1:
    nr_mini_batches_to_train = 100
elif exp_nr==2:
    nr_mini_batches_to_train = 2000
elif exp_nr==3:
    nr_mini_batches_to_train = 1000
save_filename = 'save/model.ckpt'
logs_path = './logfiles'
dropout = 0.8  # dropout (probability to keep units)


_weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, nr_img_channels, 10]), name="wc1"),
    'wc2': tf.Variable(tf.random_normal([5, 5, 10, 15]), name="wc2"),
    'wc3': tf.Variable(tf.random_normal([3, 3, 15, 20]), name="wc3"),
    'wc4': tf.Variable(tf.random_normal([3, 3, 20, 25]), name="wc4"),
    'wc5': tf.Variable(tf.random_normal([3, 3, 25, 30]), name="wc5"),
    'wd1': tf.Variable(tf.random_normal([6 * 6 * 30, 40]), name="wd1"),
    'wd2': tf.Variable(tf.random_normal([40, 40]), name="wd2"),
    'out': tf.Variable(tf.random_normal([40, n_classes]), name="wout"),
    'exp3_wd1': tf.Variable(tf.random_normal([27 * 27 * 10, 40]), name="exp3_wd1")
}

# prepare vectors for biases
_biases = {
    'bc1': tf.Variable(tf.random_normal([10]), name="bc1"),
    'bc2': tf.Variable(tf.random_normal([15]), name="bc2"),
    'bc3': tf.Variable(tf.random_normal([20]), name="bc3"),
    'bc4': tf.Variable(tf.random_normal([25]), name="bc4"),
    'bc5': tf.Variable(tf.random_normal([30]), name="bc5"),
    'bd1': tf.Variable(tf.random_normal([40]), name="bd1"),
    'bd2': tf.Variable(tf.random_normal([40]), name="bd2"),
    'bout': tf.Variable(tf.random_normal([n_classes]), name="bout")
    }



# helper function to build 1st conv layer with filter size 11x11
# and stride 4 (in both directions) and no padding
def conv1st(name, l_input, filter, b):
    with tf.name_scope(name):
        conv_op = tf.nn.conv2d(l_input, filter, strides=[1, 4, 4, 1], padding='VALID')
        relu_op = tf.nn.relu(tf.nn.bias_add(conv_op, b), name=name)
        return relu_op


# in all other layers we use a stride of 1 (in both directions)
# and a padding such that the spatial dimension (width,height)
# of the output volume is the same as the spatial dimension
# of the input volume
def conv2d(name, l_input, w, b):
    with tf.name_scope(name):
        conv_op = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
        relu_op = tf.nn.relu(tf.nn.bias_add(conv_op, b), name=name)
        return relu_op

# generates a max pooling layer
def max_pool(name, l_input, k, s):
    with tf.name_scope(name):
        pool_op = tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)
        return pool_op



# helper function to generate a CNN
def build_cnn_model(_X, keep_prob, imagesize):
    # prepare matrices for weights


    # reshape input picture
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, nr_img_channels])

    if (exp_nr == 1 or exp_nr==2):
        # topology: INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
        with tf.name_scope("TheFeatureHierarchy"):
            conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])
            pool1 = max_pool('pool1', conv1, k=3, s=2)
            conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'])
            pool2 = max_pool('pool2', conv2, k=3, s=2)
            conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'])
            conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
            conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
            pool3 = max_pool('pool3', conv5, k=3, s=2)

        with tf.name_scope("TheMLP"):
            # fully connected layer
            dense1 = tf.reshape(pool3, [-1, _weights['wd1'].get_shape().as_list()[0]])
            dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
            dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
            # dense2 = tf.nn.dropout(dense1, keep_prob)
            out = tf.matmul(dense2, _weights['out']) + _biases['bout']

    elif exp_nr == 3:

        # topology: INPUT->C1->P1->FC1->FC2->OUT
        with tf.name_scope("TheFeatureHierarchy"):
            conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])
            print("conv1 shape: ", conv1.get_shape())
            pool1 = max_pool('pool1', conv1, k=3, s=2)
            print("pool1 shape: ", pool1.get_shape())

        with tf.name_scope("TheMLP"):
            # fully connected layer
            dense1 = tf.reshape(pool1, [-1, 27*27*10])
            dense1 = tf.nn.relu(tf.matmul(dense1, _weights['exp3_wd1']) + _biases['bd1'], name='fc1')
            print("dense1 shape: ", dense1.get_shape())
            dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
            print("dense2 shape: ", dense2.get_shape())

            out = tf.matmul(dense2, _weights['out']) + _biases['bout']
            print("out shape: ", out.get_shape())

    return [out, pool1, conv1, _weights['wc1']]


"""
From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""
def log_scalar(filewriter, tag, value, step):
    """Log a scalar variable.
    Parameter
    ----------
    tag : basestring
        Name of the scalar
    value
    step : int
        training iteration
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                 simple_value=value)])

    filewriter.add_summary(summary, step)





# 1. create a training and testing Dataset object that stores
#    the training / testing images
dataset_root = "V:/01_job/12_datasets/imagenet/cars_vs_bikes_prepared/"
training = dataset(dataset_root + "train", ".jpeg", nr_img_channels)
testing = dataset(dataset_root + "validation", ".jpeg", nr_img_channels)



x = tf.placeholder(tf.float32, [None, imagesize, imagesize, nr_img_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

pred_op, pool1_op, conv1_op, weights_filter1st = build_cnn_model(x, keep_prob, imagesize)

cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_op, labels=y))
# cost = tf.reduce_mean(tf.squared_difference(pred, y))

global_step = tf.Variable(0, trainable=False)

optimizer_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost_op, global_step=global_step)
# optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost, global_step=global_step)

correct_pred_op = tf.equal(tf.argmax(pred_op, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred_op, tf.float32))

saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred_op)
tf.add_to_collection("accuracy", accuracy_op)
tf.add_to_collection("weights_filter1st", weights_filter1st)

print("\n\n")
print("----------------------------------------")
print("I am ready to start the training...")
print("So I will train a CNN, starting with a learn rate of", learn_rate)
print("I will train ", nr_mini_batches_to_train, "mini batches of ", batch_size, "images")
print("Your input images will be resized to ", imagesize, "x", imagesize, "pixels")
print("----------------------------------------")

# Create a new summary op
tf.summary.histogram('histogramconv1values', conv1_op)

# Create an "op" that will evaluate
# all summary ops in the graph at once
merged_summary_op = tf.summary.merge_all()


with tf.Session() as my_session:
    my_session.run(tf.global_variables_initializer())

    # create a file writer object
    # in order to save the summary files (log information)
    fw = tf.summary.FileWriter("V:/tmp/summary", my_session.graph)

    mini_batch_nr = 1
    while mini_batch_nr < nr_mini_batches_to_train:

        print("Training graph with mini-batch nr", mini_batch_nr,
              "of size", batch_size)

        batch_ys, batch_xs = training.nextBatch(batch_size)
        # note: batch_ys and batch_xs are tuples each
        # batch_ys a tuple of e.g. 32 one-hot NumPy arrays
        # batch_xs a tuple of e.g. 32 NumPy arrays of shape
        #  (width, height, 3)


        _, pool1_filter_values, conv1_filter_values, weights_numpyarray =\
            my_session.run([optimizer_op, pool1_op, conv1_op, weights_filter1st],
                                    feed_dict={x: batch_xs,
                                               y: batch_ys,
                                               keep_prob: dropout})

        #print("Type of conv1_filter_values is", type(conv1_filter_values))
        #print("Shape of conv1_filter_values is", conv1_filter_values.shape)

        #print("Type of pool1_filter_values is", type(pool1_filter_values))
        #print("Shape of pool1_filter_values is", pool1_filter_values.shape)

        # from time to time:
        # test model on the test dataset
        if mini_batch_nr % 100 == 0:

            print("Evaluating classification performance of model "
                  "on test data set ...")

            test_image_nr = 0
            correct = 0
            while test_image_nr < len(testing):

                # testing_ys and testing_xs are tuples
                testing_ys, testing_xs = testing.nextBatch(1)

                # get first image and first ground truth vector
                # from the tuples
                first_img = testing_xs[0]
                first_groundtruth_vec = testing_ys[0]

                # show shape of image and ground truth vector?
                if False:
                    print("Shape of testing_xs is :", first_img.shape)
                    print("Shape of testing_ys is :", first_groundtruth_vec.shape)

                # given the input image,
                # let the CNN predict the category!
                prediction_tensor, conv1_tensor, summary =\
                    my_session.run([pred_op, conv1_op, merged_summary_op],
                                   feed_dict={x: testing_xs, keep_prob: 1.})


                # get ground truth label and
                # predicted label from output vector
                groundtruth_label = np.argmax(first_groundtruth_vec)
                predicted_label = np.argmax(prediction_tensor, 1)[0]
                if predicted_label == groundtruth_label:
                    correct += 1
                test_image_nr += 1

            # end while

            classification_rate = correct / test_image_nr * 100.0
            print("Classification rate in percent = {0:.2f}".format(classification_rate))

            # stuff to log for TensorBoard visualization

            # log classification rate
            log_scalar(fw, "classificationrate", classification_rate, mini_batch_nr)

            # log other stuff:
            # - conv1 tensor, i.e. outputs of conv1 neurons
            # - images of filters in 1st layer
            fw.add_summary(summary, mini_batch_nr)

        # end if (do new evaluation on test data set)

        mini_batch_nr += 1

    # while mini_batch_nr < nr_mini_batches_to_train:

    print("\n")
    print("Training of CNN model finished.")

    save_filename = "saved_model_exp0" + str(exp_nr) + "/final_model.ckpt"
    saver.save(my_session, save_filename, global_step=mini_batch_nr)
    print("Saved CNN model to file '",save_filename,"'")
