import pandas as pd              # A beautiful library to help us work with data as tables
import numpy as np               # So we can use number matrices. Both pandas and TensorFlow need it.
import tensorflow as tf          # Fire from the gods
import matplotlib.pyplot as plt  # Visualize the things
import matplotlib.image as img
import os

def guess(input_file):
    image = img.imread(input_file)
    image = np.asarray(image)
    gray = rgb2gray(image)
    gray.resize((100,100))
    flatten= gray.flatten()
    input_test= np.reshape(flatten,(1,10000))
    print(sess.run(y, feed_dict={x: input_test}))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

n_pixels = 10000 # 100x100
n_classes = 2
num_examples = 138
df = pd.read_csv('dataset.csv')
df.columns = ['y1', 'x']

df.insert(1, "y2", df["y1"] == 0)
df.loc[:, ("y2")] = df["y2"].astype(int)


inputY = df.loc[:, ["y1", "y2"]].as_matrix()

inputX = np.zeros((num_examples, 10000),dtype=np.float32)
for i in range(inputY.shape[0]):
    inputX[i] = np.fromstring(df.loc[:,"x"].loc[i],dtype=np.float32,sep=' ')


# Parameters
learning_rate = 0.0001
training_epochs = 100000
display_step = 50
n_samples = inputY.size

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

hidden_l1 = {
    'weights':tf.Variable(tf.random_normal([n_pixels, n_nodes_hl1],dtype=tf.float32)),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1],dtype=tf.float32))
}
hidden_l2 = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2],dtype=tf.float32)),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2],dtype=tf.float32))
}
hidden_l3 = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3],dtype=tf.float32)),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3],dtype=tf.float32))
}
output_layer = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes],dtype=tf.float32)),
    'biases':tf.Variable(tf.random_normal([n_classes],dtype=tf.float32))
}


x = tf.placeholder(tf.float32, [None, 10000])
y_ = tf.placeholder(tf.float32, [None, n_classes])

# W = tf.Variable(tf.zeros([n_pixels, n_classes],dtype=tf.float32))
# b = tf.Variable(tf.zeros([n_classes],dtype=tf.float32))
# y_values = tf.add(tf.matmul(x, W), b)

l1 = tf.add(tf.matmul(x, hidden_l1['weights']), hidden_l1['biases'])
l1 = tf.nn.softmax(l1)

l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
l2 = tf.nn.softmax(l2)

l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
l3 = tf.nn.softmax(l3)

output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
print(output)
y = tf.nn.softmax(output)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_) )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initialize variabls and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    # That's all! The rest of the cell just outputs debug messages.
    # Display logs per epoch step
    if (i) % display_step == 0:
        _, c = sess.run([optimizer, cost], feed_dict={x: inputX, y_: inputY})
        print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(c)) #, \"W=", sess.run(W), "b=", sess.run(b)
        guess("test_apple1.jpg")
        guess("test_apple2.jpg")
        guess("test_bananas1.jpg")
        guess("test_bananas2.jpg")
print("Optimization Finished!")
_, c = sess.run([optimizer, cost], feed_dict={x: inputX, y_: inputY})
print("Training cost=", c)






