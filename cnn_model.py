import numpy as np
import tensorflow as tf
from collections import namedtuple
import random

HParams = namedtuple('HParams',
                     'batch_size, min_lrn_rate, lrn_rate, '
                     'weight_decay_rate, '
                     ' optimizer')
FLAGS = tf.app.flags.FLAGS


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

def defModel():
  # in conv1 pool1 conv2 pool2 h1 o
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.int64)
  x_2D = tf.reshape(X,[-1,28,28,1])
  #init vars
  drop_out_input = tf.placeholder(tf.float32)
  drop_out_hidden = tf.placeholder(tf.float32)

  with tf.name_scope("drop_out"):
    x_2D = tf.nn.dropout(x_2D,drop_out_input)

  with tf.name_scope("conv1"):
    W_conv1 = init_weights(shape = [5,5,1,32], name = "W_conv1")
    b_conv1 = init_weights(shape = [32], name = "b_conv1")
    conv1 = tf.nn.conv2d(x_2D,W_conv1,strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1+b_conv1)
  with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(conv1,ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

  with tf.name_scope("conv2"):
    W_conv2 = init_weights(shape = [5,5,32,64], name = "W_conv2")
    b_conv2 = init_weights(shape = [64], name = "b_conv2")
    conv2 = tf.nn.conv2d(pool1,W_conv2,strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2+b_conv2)
  with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2,ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

  with tf.name_scope("hidden"):
    W_fc1 = init_weights(shape = [7*7*64,512], name = "W_fc1")
    b_fc1 = init_weights(shape = [512], name = "b_fc1")
    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(tensor = pool2,shape =[-1,  7*7*64],name="reshape_pool")
    hidden = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
    hidden = tf.nn.dropout(hidden,drop_out_hidden)
  
  with tf.name_scope("out"):
    W_fc2 = init_weights(shape = [512,10], name = "W_fc2")
    b_fc2 = init_weights(shape = [10], name = "b_fc2")
    logits = tf.matmul(hidden,W_fc2) + b_fc2
    l2_loss = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(b_fc2)
  return X,Y,logits,l2_loss,drop_out_input,drop_out_hidden

def getTrainOps():
  X,Y,logits,l2_loss,drop_out_input,drop_out_hidden = defModel()
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y, logits=logits))
    tf.summary.scalar('loss/loss_cross_entropy',loss)
    loss += 5e-4 * l2_loss
    tf.summary.scalar('loss/l2_loss',l2_loss)
  with tf.name_scope('precision_log'):
    tf.summary.scalar('precision', tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y), tf.float32)))
  return X,Y,tf.train.AdamOptimizer(1e-4).minimize(loss) ,drop_out_input,drop_out_hidden

def getEvalOps():
  X,Y,logits,l2_loss,drop_out_input,drop_out_hidden = defModel()
  return X,Y,tf.argmax(logits,1),drop_out_input,drop_out_hidden



class LeNet(object):
  """docstring for LeNet"""
  def __init__(self, hps):
      super(LeNet, self).__init__()
      self.hps = hps
  
  def train(self,train_data):
    """Training loop."""
    X,Y,train_op,drop_out_input,drop_out_hidden = getTrainOps()
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      logDirPath = FLAGS.log_root
      writer = tf.summary.FileWriter(logDirPath,sess.graph)
      saver = tf.train.Saver()
      merged = tf.summary.merge_all()
      sess.run(init)
      print self.hps.batch_size
      for i in range(1):
        batch = random.sample(train_data[:-2*self.hps.batch_size],self.hps.batch_size)
        images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
        sess.run(train_op,feed_dict={X:images,Y:labels,drop_out_input:0.5,drop_out_hidden:0.6})
        if i%100 == 0:
          batch = train_data[-2*self.hps.batch_size:]
          images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
          result = sess.run(merged, feed_dict={X:images,Y:labels,drop_out_input:1,drop_out_hidden:1})
          writer.add_summary(result,i)
      saver.save(sess,FLAGS.model_param_path)
      writer.close()
  def eval(self,eval_data):
    X,Y,evalResult,drop_out_input,drop_out_hidden = getEvalOps()
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess,FLAGS.model_param_path)
      loopCount = len(eval_data)//self.hps.batch_size
      result = []
      for i in range(loopCount+1):
        batch = eval_data[i*self.hps.batch_size:(i+1)*self.hps.batch_size]
        images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
        result+=sess.run(evalResult,feed_dict={X:images,Y:labels,drop_out_input:1,drop_out_hidden:1}).tolist()
    return result




