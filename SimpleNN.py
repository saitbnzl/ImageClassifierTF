import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.image as img
import os
import sys

if "-floyd" in sys.argv:
    output_path = "/output/"
else:
    output_path = "./output/"

# Parameters
n_pixels = 10000 # 100x100
n_classes = 2
num_examples = 138
learning_rate = 0.0001
training_epochs = 10000
display_step = 50

x = tf.placeholder(tf.float32, [None, 10000])
y = tf.Variable(tf.zeros([1,2],dtype=tf.float32))
y_ = tf.placeholder(tf.float32, [None, n_classes])

def loadModel():
    if os.path.isfile(output_path+"model.ckpt.meta"):
        saver = tf.train.import_meta_graph(output_path+'model.ckpt.meta')
        with tf.Session() as sess:
            saver.restore(sess, output_path+"model.ckpt")
            print("Model restored...")

def load_and_guess(input_file):
    with tf.Session() as sess:
        image = img.imread(input_file)
        image = np.asarray(image)
        gray = rgb2gray(image)
        gray.resize((100,100))
        flatten= gray.flatten()
        input_test= np.reshape(flatten,(1,10000))
        y = neural_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(y, feed_dict={x: input_test}))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])





def load_dataset():
    df = pd.read_csv('dataset.csv')
    df.columns = ['y1', 'x']

    df.insert(1, "y2", df["y1"] == 0)
    df.loc[:, ("y2")] = df["y2"].astype(int)

    inputY = df.loc[:, ["y1", "y2"]].as_matrix()
    inputX = np.zeros((num_examples, 10000),dtype=np.float32)
    for i in range(inputY.shape[0]):
        inputX[i] = np.fromstring(df.loc[:,"x"].loc[i],dtype=np.float32,sep=' ')
    return inputX,inputY



def neural_network():
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500

    hidden_l1 = {
        'weights':tf.Variable(tf.random_normal([n_pixels, n_nodes_hl1],dtype=tf.float32),name="h1w"),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl1],dtype=tf.float32),name="h1b")
    }
    hidden_l2 = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2],dtype=tf.float32),name="h2w"),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl2],dtype=tf.float32),name="h2b")
    }
    hidden_l3 = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3],dtype=tf.float32),name="h3w"),
        'biases':tf.Variable(tf.random_normal([n_nodes_hl3],dtype=tf.float32),name="h3b")
    }
    output_layer = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes],dtype=tf.float32),name="ow"),
        'biases':tf.Variable(tf.random_normal([n_classes],dtype=tf.float32),name="ob")
    }

    l1 = tf.add(tf.matmul(x, hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.softmax(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.softmax(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.softmax(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    output = tf.nn.softmax(output)
    return output

def train_network():
    inputX,inputY = load_dataset()
    y = neural_network()
    # y = tf.nn.softmax(y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        if "-r" in sys.argv:
            loadModel()
        for i in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x: inputX, y_: inputY})
            if (i) % display_step == 0:
                print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(c)) #, \"W=", sess.run(W), "b=", sess.run(b)
        save_path = saver.save(sess, output_path+"model.ckpt")
        print("Model saved in file: %s" % save_path)
        print("Optimization Finished!")
        _, c = sess.run([optimizer, cost], feed_dict={x: inputX, y_: inputY})
        print("Training cost=", c)

train_network()










