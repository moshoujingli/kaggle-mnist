import numpy as np
import tensorflow as tf
import kaggle_input
import cnn_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'kaggle', 'kaggle or mnist.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('model_param_path', './model/',
                           'Path to save trained model')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', './recognize_result',
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

def train(hps):
  """Training loop."""
  train_data = kaggle_input.build_input_train(FLAGS.train_data_path)
  model = cnn_model.SimpleCNN(hps)
  model.train(train_data)

def evaluate(hps):
  pass
  eval_data = kaggle_input.build_input_eval(FLAGS.eval_data_path)
  model = cnn_model.SimpleCNN(hps)
  result = model.eval(eval_data)
  with open(FLAGS.eval_dir+'/result.csv','wb') as outFile:
    outFile.write("ImageId,Label\n")
    for index, item in enumerate(result):
      outFile.write("%d,%d\n"%(index+1,item))

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
    batch_size = 256

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