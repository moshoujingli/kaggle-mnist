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

  with tf.name_scope("hidden"):
    W_hidden = init_weights(shape = [784,1024], name = "W_hidden")
    b_hidden = init_weights(shape = [1024], name = "b_hidden")
    hidden = tf.nn.relu(tf.matmul(X,W_hidden)+b_hidden)
  
  with tf.name_scope("out"):
    W_out = init_weights(shape = [1024,10], name = "W_out")
    b_out = init_weights(shape = [10], name = "b_out")
    logits = tf.matmul(hidden,W_out) + b_out
  return X,Y,logits

def getTrainOps():
  X,Y,logits = defModel()
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y, logits=logits))
    tf.summary.scalar('loss/loss_cross_entropy',loss)
  with tf.name_scope('precision_log'):
    tf.summary.scalar('precision', tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y), tf.float32)))
  return X,Y,tf.train.AdamOptimizer(1e-4).minimize(loss)

def getEvalOps():
  X,Y,logits = defModel()
  return X,Y,tf.argmax(logits,1)



class SimpleNN(object):
  """docstring for SimpleNN"""
  def __init__(self, hps):
      super(SimpleNN, self).__init__()
      self.hps = hps
  
  def train(self,train_data):
    """Training loop."""
    X,Y,train_op = getTrainOps()
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      logDirPath = FLAGS.log_root
      writer = tf.summary.FileWriter(logDirPath,sess.graph)
      saver = tf.train.Saver()
      merged = tf.summary.merge_all()
      sess.run(init)
      print self.hps.batch_size
      for i in range(4000):
        batch = random.sample(train_data[:-2*self.hps.batch_size],self.hps.batch_size)
        images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
        sess.run(train_op,feed_dict={X:images,Y:labels})
        if i%100 == 0:
          batch = train_data[-2*self.hps.batch_size:]
          images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
          result = sess.run(merged, feed_dict={X:images,Y:labels})
          writer.add_summary(result,i)
      saver.save(sess,FLAGS.model_param_path)
      writer.close()
  def eval(self,eval_data):
    X,Y,evalResult = getEvalOps()
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess,FLAGS.model_param_path)
      loopCount = len(eval_data)//self.hps.batch_size
      result = []
      for i in range(loopCount+1):
        batch = eval_data[i*self.hps.batch_size:(i+1)*self.hps.batch_size]
        images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
        result+=sess.run(evalResult,feed_dict={X:images,Y:labels}).tolist()
    return result




