#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("image", "./data/inference/Pikachu.png",
                    "The image to inference")
flags.DEFINE_string(
    "model", "cnn",
    "Model to train, option model: cnn, lstm, bidirectional_lstm, stacked_lstm")


def main():
  print("Start Pokemon classifier")

  # Initialize train and test data
  TRAIN_IMAGE_NUMBER = 646
  TEST_IMAGE_NUMBER = 68
  IMAGE_SIZE = 32
  RGB_CHANNEL_SIZE = 3
  LABEL_SIZE = 17

  train_dataset = np.ndarray(
      shape=(TRAIN_IMAGE_NUMBER, IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNEL_SIZE),
      dtype=np.float32)
  test_dataset = np.ndarray(
      shape=(TEST_IMAGE_NUMBER, IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNEL_SIZE),
      dtype=np.float32)

  train_labels = np.ndarray(shape=(TRAIN_IMAGE_NUMBER, ), dtype=np.int32)
  test_labels = np.ndarray(shape=(TEST_IMAGE_NUMBER, ), dtype=np.int32)

  TRAIN_DATA_DIR = "./data/train/"
  TEST_DATA_DIR = "./data/test/"
  VALIDATE_DATA_DIR = "./data/validate/"
  IMAGE_FORMAT = ".png"
  index = 0
  pokemon_type_id_map = {"Bug": 0,
                         "Dark": 1,
                         "Dragon": 2,
                         "Electric": 3,
                         "Fairy": 4,
                         "Fighting": 5,
                         "Fire": 6,
                         "Ghost": 7,
                         "Grass": 8,
                         "Ground": 9,
                         "Ice": 10,
                         "Normal": 11,
                         "Poison": 12,
                         "Psychic": 13,
                         "Rock": 14,
                         "Steel": 15,
                         "Water": 16}
  pokemon_types = ["Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting",
                   "Fire", "Ghost", "Grass", "Ground", "Ice", "Normal",
                   "Poison", "Psychic", "Rock", "Steel", "Water"]

  # Load train images
  for pokemon_type in os.listdir(TRAIN_DATA_DIR):
    for image_filename in os.listdir(os.path.join(TRAIN_DATA_DIR,
                                                  pokemon_type)):
      if image_filename.endswith(IMAGE_FORMAT):

        image_filepath = os.path.join(TRAIN_DATA_DIR, pokemon_type,
                                      image_filename)
        image_ndarray = ndimage.imread(image_filepath, mode="RGB")
        train_dataset[index] = image_ndarray

        train_labels[index] = pokemon_type_id_map.get(pokemon_type)
        index += 1

  index = 0
  # Load test image
  for pokemon_type in os.listdir(TEST_DATA_DIR):
    for image_filename in os.listdir(os.path.join(TEST_DATA_DIR,
                                                  pokemon_type)):
      if image_filename.endswith(IMAGE_FORMAT):

        image_filepath = os.path.join(TEST_DATA_DIR, pokemon_type,
                                      image_filename)
        image_ndarray = ndimage.imread(image_filepath, mode="RGB")
        test_dataset[index] = image_ndarray

        test_labels[index] = pokemon_type_id_map.get(pokemon_type)
        index += 1

  # Define the model
  x = tf.placeholder(tf.float32,
                     shape=(None, IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNEL_SIZE))
  y = tf.placeholder(tf.int32, shape=(None, ))

  batch_size = FLAGS.batch_size
  epoch_number = FLAGS.epoch_number
  checkpoint_dir = FLAGS.checkpoint_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  tensorboard_dir = FLAGS.tensorboard_dir
  mode = FLAGS.mode
  checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
  steps_to_validate = FLAGS.steps_to_validate

  def cnn_inference(x):
    # Convolution layer result: [BATCH_SIZE, 16, 16, 64]
    with tf.variable_scope("conv1"):
      weights = tf.get_variable("weights",
                                [3, 3, 3, 32],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [32],
                             initializer=tf.random_normal_initializer())

      layer = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)
      layer = tf.nn.max_pool(layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")

    # Convolution layer result: [BATCH_SIZE, 8, 8, 64]
    with tf.variable_scope("conv2"):
      weights = tf.get_variable("weights",
                                [3, 3, 32, 64],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [64],
                             initializer=tf.random_normal_initializer())

      layer = tf.nn.conv2d(layer,
                           weights,
                           strides=[1, 1, 1, 1],
                           padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)
      layer = tf.nn.max_pool(layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")

    # Reshape for full-connect network
    layer = tf.reshape(layer, [-1, 8 * 8 * 64])
    #import ipdb;ipdb.set_trace()

    # Full connected layer result: [BATCH_SIZE, 17]
    with tf.variable_scope("fc1"):
      # weights.get_shape().as_list()[0]] = 8 * 8 * 64
      weights = tf.get_variable("weights",
                                [8 * 8 * 64, LABEL_SIZE],
                                initializer=tf.random_normal_initializer())
      bias = tf.get_variable("bias",
                             [LABEL_SIZE],
                             initializer=tf.random_normal_initializer())
      layer = tf.add(tf.matmul(layer, weights), bias)

    return layer

  def lstm_inference(x):
    RNN_HIDDEN_UNITS = 128

    # x was [BATCH_SIZE, 32, 32, 3]
    # x changes to [32, BATCH_SIZE, 32, 3]
    x = tf.transpose(x, [1, 0, 2, 3])
    # x changes to [32 * BATCH_SIZE, 32 * 3]
    x = tf.reshape(x, [-1, IMAGE_SIZE * RGB_CHANNEL_SIZE])
    # x changes to array of 32 * [BATCH_SIZE, 32 * 3]
    x = tf.split(0, IMAGE_SIZE, x)

    weights = tf.Variable(tf.random_normal([RNN_HIDDEN_UNITS, LABEL_SIZE]))
    biases = tf.Variable(tf.random_normal([LABEL_SIZE]))

    # output size is 128, state size is (c=128, h=128)
    lstm_cell = rnn_cell.BasicLSTMCell(RNN_HIDDEN_UNITS, forget_bias=1.0)
    # outputs is array of 32 * [BATCH_SIZE, 128]
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # outputs[-1] is [BATCH_SIZE, 128]
    return tf.matmul(outputs[-1], weights) + biases

  def bidirectional_lstm_inference(x):
    RNN_HIDDEN_UNITS = 128

    # x was [BATCH_SIZE, 32, 32, 3]
    # x changes to [32, BATCH_SIZE, 32, 3]
    x = tf.transpose(x, [1, 0, 2, 3])
    # x changes to [32 * BATCH_SIZE, 32 * 3]
    x = tf.reshape(x, [-1, IMAGE_SIZE * RGB_CHANNEL_SIZE])
    # x changes to array of 32 * [BATCH_SIZE, 32 * 3]
    x = tf.split(0, IMAGE_SIZE, x)

    weights = tf.Variable(tf.random_normal([2 * RNN_HIDDEN_UNITS, LABEL_SIZE]))
    biases = tf.Variable(tf.random_normal([LABEL_SIZE]))

    # output size is 128, state size is (c=128, h=128)
    fw_lstm_cell = rnn_cell.BasicLSTMCell(RNN_HIDDEN_UNITS, forget_bias=1.0)
    bw_lstm_cell = rnn_cell.BasicLSTMCell(RNN_HIDDEN_UNITS, forget_bias=1.0)
    # outputs is array of 32 * [BATCH_SIZE, 128]
    outputs, _, _ = rnn.bidirectional_rnn(fw_lstm_cell,
                                          bw_lstm_cell,
                                          x,
                                          dtype=tf.float32)

    # outputs[-1] is [BATCH_SIZE, 128]
    return tf.matmul(outputs[-1], weights) + biases

  def stacked_lstm_inference(x):
    RNN_HIDDEN_UNITS = 128

    # x was [BATCH_SIZE, 32, 32, 3]
    # x changes to [32, BATCH_SIZE, 32, 3]
    x = tf.transpose(x, [1, 0, 2, 3])
    # x changes to [32 * BATCH_SIZE, 32 * 3]
    x = tf.reshape(x, [-1, IMAGE_SIZE * RGB_CHANNEL_SIZE])
    # x changes to array of 32 * [BATCH_SIZE, 32 * 3]
    x = tf.split(0, IMAGE_SIZE, x)

    weights = tf.Variable(tf.random_normal([RNN_HIDDEN_UNITS, LABEL_SIZE]))
    biases = tf.Variable(tf.random_normal([LABEL_SIZE]))

    # output size is 128, state size is (c=128, h=128)
    lstm_cell = rnn_cell.BasicLSTMCell(RNN_HIDDEN_UNITS, forget_bias=1.0)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    # outputs is array of 32 * [BATCH_SIZE, 128]
    outputs, states = rnn.rnn(lstm_cells, x, dtype=tf.float32)

    # outputs[-1] is [BATCH_SIZE, 128]
    return tf.matmul(outputs[-1], weights) + biases

  def inference(inputs):
    print("Use the model: {}".format(FLAGS.model))
    if FLAGS.model == "cnn":
      return cnn_inference(inputs)
    elif FLAGS.model == "lstm":
      return lstm_inference(inputs)
    elif FLAGS.model == "bidirectional_lstm":
      return bidirectional_lstm_inference(inputs)
    elif FLAGS.model == "stacked_lstm":
      return stacked_lstm_inference(inputs)
    else:
      print("Unknow model, exit now")
      exit(1)

  # Define train op
  logit = inference(x)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit,
                                                                       y))

  learning_rate = FLAGS.learning_rate
  print("Use the optimizer: {}".format(FLAGS.optimizer))
  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
    exit(1)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  # Define accuracy and inference op
  tf.get_variable_scope().reuse_variables()
  logits = inference(x)
  validate_softmax = tf.nn.softmax(logits)
  predict_op = tf.argmax(validate_softmax, 1)
  correct_prediction = tf.equal(predict_op, tf.to_int64(y))
  accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  tf.scalar_summary('loss', loss)
  init_op = tf.initialize_all_variables()

  # Create session to run graph
  with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
    sess.run(init_op)

    if mode == "train":
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Continue training from the model {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      start_time = datetime.datetime.now()
      for epoch in range(epoch_number):

        _, loss_value, step = sess.run(
            [train_op, loss, global_step],
            feed_dict={x: train_dataset,
                       y: train_labels})

        if epoch % steps_to_validate == 0:
          end_time = datetime.datetime.now()

          train_accuracy_value, summary_value = sess.run(
              [accuracy_op, summary_op],
              feed_dict={x: train_dataset,
                         y: train_labels})
          test_accuracy_value = sess.run(accuracy_op,
                                         feed_dict={x: test_dataset,
                                                    y: test_labels})

          print(
              "[{}] Epoch: {}, loss: {}, train_accuracy: {}, test_accuracy: {}".format(
                  end_time - start_time, epoch, loss_value,
                  train_accuracy_value, test_accuracy_value))

          saver.save(sess, checkpoint_file, global_step=step)
          writer.add_summary(summary_value, step)
          start_time = end_time

    elif mode == "inference":
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Load the model {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      start_time = datetime.datetime.now()

      image_ndarray = ndimage.imread(FLAGS.image, mode="RGB")
      print_image(image_ndarray)

      image_ndarray = image_ndarray.reshape(1, IMAGE_SIZE, IMAGE_SIZE,
                                            RGB_CHANNEL_SIZE)
      prediction = sess.run(predict_op, feed_dict={x: image_ndarray})

      end_time = datetime.datetime.now()
      pokemon_type = pokemon_types[prediction[0]]
      print("[{}] Predict type: {}".format(end_time - start_time,
                                           pokemon_type))

    else:
      print("Unknow mode, please choose 'train' or 'inference'")

  print("End of Pokemon classifier")


def print_image(image_ndarray):
  plt.imshow(image_ndarray)
  plt.show()


if __name__ == "__main__":
  main()
