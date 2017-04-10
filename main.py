import numpy as np
import tensorflow as tf
import kaggle_input
import random
import cnn_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'kaggle', 'kaggle or mnist.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_integer('batch_size_assigned', -1,
                            'Number of batche size assigned.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', './tensor_board_data',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

def defModel():
  # in conv1 pool1 conv2 pool2 h1 o
  X = tf.placeholder("float", [None, 28, 28, 1])
  Y = tf.placeholder(tf.int64)
  #init vars
  W_conv1 = init_weights(shape = [5,5,1,32], name = "W_conv1")
  b_conv1 = init_weights(shape = [32], name = "b_conv1")

  W_conv2 = init_weights(shape = [5,5,32,64], name = "W_conv2")
  b_conv2 = init_weights(shape = [64], name = "b_conv2")

  W_fc1 = init_weights(shape = [7*7*64,512], name = "W_fc1")
  b_fc1 = init_weights(shape = [512], name = "b_fc1")

  W_fc2 = init_weights(shape = [512,10], name = "W_fc2")
  b_fc2 = init_weights(shape = [10], name = "b_fc2")


  conv1 = tf.nn.conv2d(X,W_conv1,strides=[1, 1, 1, 1], padding='SAME')
  conv1 = tf.nn.relu(conv1+b_conv1)
  pool1 = tf.nn.max_pool(conv1,ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

  conv2 = tf.nn.conv2d(pool1,W_conv2,strides=[1, 1, 1, 1], padding='SAME')
  conv2 = tf.nn.relu(conv2+b_conv2)
  pool2 = tf.nn.max_pool(conv2,ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
  pool_shape = pool2.get_shape().as_list()
  reshape = tf.reshape(tensor = pool2,shape =[-1,  7*7*64],name="reshape_pool")

  hidden = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
  logits = tf.matmul(hidden,W_fc2) + b_fc2

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=Y, logits=logits))
  tf.summary.scalar('loss/loss_cross_entropy',loss)

  l2_loss = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(b_fc2)
  loss += 5e-4 * l2_loss
  tf.summary.scalar('loss/l2_loss',l2_loss)
  tf.summary.scalar('precision', tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y), tf.float32)))


  return X,Y,tf.train.AdamOptimizer(1e-4).minimize(loss)

def train(hps):
  """Training loop."""
  train_data = kaggle_input.build_input(FLAGS.train_data_path)
  X,Y,train_op = defModel()
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    logDirPath = FLAGS.log_root
    writer = tf.summary.FileWriter(logDirPath,sess.graph)
    merged = tf.summary.merge_all()
    sess.run(init)
    print hps.batch_size
    for i in range(4000):
      batch = random.sample(train_data,hps.batch_size)
      images,labels = np.array([img for label,img in batch]),np.array([label for label,img in batch])
      sess.run(train_op,feed_dict={X:images,Y:labels})
      if i%100 == 0:
        result = sess.run(merged, feed_dict={X: images, Y: labels})
        writer.add_summary(result,i)
    writer.close()

def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.batch_size_assigned >0:
    batch_size = FLAGS.batch_size_assigned

  hps = cnn_model.HParams(batch_size=batch_size,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             weight_decay_rate=0.0002,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()