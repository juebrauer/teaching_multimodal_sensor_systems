# A first simple TensorFlow learner
#
# Given some random samples of 2D data points (x,y),
# the following TF script tries to find a line
# that best describes the data points in terms of
# a minimal sum of squared errors (SSE)
#
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org

import tensorflow as tf
import numpy as np

# 1.1 create 1D array with 100 random numbers drawn uniformly from [0,1)
x_data = np.random.rand(100)
print("\nx_data:", x_data)
print("data type:", type(x_data))

# 1.2 now compute the y-points
y_data = x_data * 1.2345 + 0.6789

# so now we have ground truth samples (x,y)
# and the TF learner will have to estimate the line parameters
# y=W*x+b with W=1.2345 and b=0.6789
#
# These parameters are called variables in TF.


# 2. We initialize them with a random W and b=0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 3.1 Now define what to optimize at all:
#     here we want to minimize the SSE.
#     This is our "loss" of a certain line model
loss_func = tf.reduce_mean(tf.square(y - y_data))

# 3.2 We can use different optimizers in TF for model learning
my_optimizer = tf.train.GradientDescentOptimizer(0.5)

# 3.3 Tell the optimizer object to minimize the loss function
train = my_optimizer.minimize(loss_func)

# 4. Before starting, we have to initialize the variables
init = tf.global_variables_initializer()

# 5. Now distribute the graph to the computing hardware
sess = tf.Session()

# 6. Initialize the graph
sess.run(init)

# 7. Print inial value of W and b
print("\n")
print("initial W", sess.run(W))
print("initial b", sess.run(b))

# 8. For 200 steps...
print("\n")
for step in range(201):

    # 8.1 Do another gradient descent step to come to a better
    #     W and b
    sess.run(train)

    # 8.2 From time to time, print the current value of W and b
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b))