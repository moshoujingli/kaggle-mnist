import numpy as np
import tensorflow as tf
from collections import namedtuple
import random
# 0.97214
HParams = namedtuple('HParams',
                     'iter_count,batch_size, min_lrn_rate, lrn_rate, '
                     'weight_decay_rate, '
                     ' optimizer')
FLAGS = tf.app.flags.FLAGS


def init_weights(shape, name):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)





class SimpleRNN(object):
  """docstring for SimpleNN"""
  def __init__(self, hps):
      super(SimpleRNN, self).__init__()
      self.hps = hps
  def RNN(self,X):
    n_inputs = 28
    n_steps = 28
    n_hidden_units = 128
    weights={'in':init_weights([28,128],'w_in'),'out':init_weights([128,10],'w_out')}
    biases={'in':init_weights([128],'b_in'),'out':init_weights([10],'b_out')}
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(self.hps.batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state= init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

  def defModel(self):
    # in conv1 pool1 conv2 pool2 h1 o
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int64)

    reshaped = tf.reshape(X,[-1,28,28])
    logits = self.RNN(reshaped)
    return X,Y,logits

  def getTrainOps(self):
    X,Y,logits = self.defModel()
    with tf.name_scope("loss"):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=Y, logits=logits))
      tf.summary.scalar('loss/loss_cross_entropy',loss)
    with tf.name_scope('precision_log'):
      tf.summary.scalar('precision', tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y), tf.float32)))
    return X,Y,tf.train.AdamOptimizer(1e-4).minimize(loss)

  def getEvalOps(self):
    X,Y,logits = self.defModel()
    return X,Y,tf.argmax(logits,1)

  def train(self,train_data):
    """Training loop."""
    X,Y,train_op = self.getTrainOps()
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      logDirPath = FLAGS.log_root
      writer = tf.summary.FileWriter(logDirPath,sess.graph)
      saver = tf.train.Saver()
      merged = tf.summary.merge_all()
      sess.run(init)
      print self.hps.batch_size
      for i in range(self.hps.iter_count):
        batch = random.sample(train_data[:-2*self.hps.batch_size],self.hps.batch_size)
        images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
        sess.run(train_op,feed_dict={X:images,Y:labels})
        if i%100 == 0:
          batch = train_data[-1*self.hps.batch_size:]
          images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
          result = sess.run(merged, feed_dict={X:images,Y:labels})
          writer.add_summary(result,i)
      saver.save(sess,FLAGS.model_param_path)
      writer.close()
  def eval(self,eval_data):
    X,Y,evalResult = self.getEvalOps()
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




