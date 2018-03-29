# 1 数据yale 2 准备train label-》train 
# 3 cnn 4 检测
import tensorflow as tf
import numpy as np
import scipy.io as sio
# 设置了gpu加速提示信息太多了，设置日志等级屏蔽一些
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

f = open('Yale_64x64.mat','rb')
mdict = sio.loadmat(f)

# fea gnd; key value
train_data = mdict['fea']
train_label = mdict['gnd']

# 实现数据的无序排列
train_data = np.random.permutation(train_data)
train_label = np.random.permutation(train_label)

test_data = train_data[0:64]
test_label = train_label[0:64]

# 设置随机种子
np.random.seed(100)
test_data = np.random.permutation(test_data)
# 对齐
np.random.seed(100)
test_label = np.random.permutation(test_label)

# train [0-9] [10*N] [15*N]  onehot [0 0 1 0 0 0 0 0 0 0] -> 2
# traindata进行处理 / 255完成归一化
train_data = train_data.reshape(train_data.shape[0],64,64,1).astype(np.float32)/255
train_labels_new = np.zeros((165,15))# 165 image 15人
for i in range(0,165):
    j = int(train_label[i,0])-1 # 1-15 转换为 0-14 
    train_labels_new[i,j] = 1

test_data_input = test_data.reshape(test_data.shape[0],64,64,1).astype(np.float32)/255
test_labels_input = np.zeros((64,15))# 165 image 15
for i in range(0,64):
    j = int(test_label[i,0])-1 # 1-15 0-14 
    test_labels_input[i,j] = 1
    
# cnn acc  tf.nn tf.layer
data_input = tf.placeholder(tf.float32,[None,64,64,1])
label_input = tf.placeholder(tf.float32,[None,15])

layer1 = tf.layers.conv2d(inputs=data_input,filters=32,kernel_size=2,strides=1,padding='SAME',activation=tf.nn.relu)
layer1_pool = tf.layers.max_pooling2d(layer1,pool_size=2,strides=2)
layer2 = tf.reshape(layer1_pool,[-1,32*32*32])
layer2_relu = tf.layers.dense(layer2,1024,tf.nn.relu)
output = tf.layers.dense(layer2_relu,15)

loss = tf.losses.softmax_cross_entropy(onehot_labels=label_input,logits=output)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(label_input,axis=1),predictions=tf.argmax(output,axis=1))[1]

# run acc
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,200):
        train_data_input = np.array(train_data)
        train_label_input = np.array(train_labels_new)
        sess.run([train,loss],feed_dict={data_input:train_data_input,label_input:train_label_input})
        acc = sess.run(accuracy,feed_dict={data_input:test_data_input,label_input:test_labels_input})
        print('acc:%.2f',acc)