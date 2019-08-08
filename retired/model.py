# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""
import sys


from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


import logging

import numpy as np

def _HERE(*args):
  """Helper function for getting the current directory of this script."""
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))


LearningRate = 0.0010000000 
Momentum = 0.9000000000 
Regularization = 0.0000100000 

import os
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
print(tf.__version__)
#sys.path.append("./python_packages")
sys.path.append(_HERE()+"/efficientnet")
#sys.path.append(_HERE()+"/autoaugment")
#sys.path.append(_HERE()+"/res18")
from efficientnet import EfficientNet
#from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
#from resnet import ResNet

import threading


threads = [threading.Thread(target=lambda: tf.Session())]
[t.start() for t in threads]

class Model(object):
  """Trivial example of valid model. Returns all-zero predictions."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """

    self.number_of_labels = metadata.get_output_size()
    self.input_shape = metadata.get_tensor_size()

    self.done_training = False
    self.metadata = metadata

    self.train_iterator = None
    self.train_iterator_handle = None
    self.test_iterator = None
    self.test_iterator_handle = None

    self.train_data = None
    self.train_labels = None
    self.test_data = None

    self.augment_policy = None

    self.train_count = 0
    self.FirstIteration = True
    self.learning_rate_patience = 100
    self.weight_decay_factor = 0.8
    self.best_loss = 1E10

    self.init_batch_size = 32
    self.batch_size = 64
    self.max_edge = 64
    self.min_edge = 32
    self.default_factor = 0.5

    self.lr = 0.002
    self.wd = 0.000025
    self.fc_size = 256
    self.fc_bottleneck = True
    self.network_code = 'efficientnet-bmini'
    self.pretraining = "imagenet"
    #self.network_code = 'efficient net-b0'
    self.dropout_rate = 0.2

    self.do_multilabel_check = True
    self.adaptive_loss = False

    # text_file = open("config.txt",  "r")
    # conf = text_file.read()
    # text_file.close()
    # conf = conf.split("_")
    # self.lr = float(conf[0])
    # self.learning_rate_patience = int(conf[1])
    # self.weight_decay_factor = float(conf[2])
    # self.wd = float(conf[3])
    # self.max_edge = int(conf[4])
    # self.fc_size = int(conf[5])
    # self.network_code = conf[6]
    # self.pretraining = conf[7]
    # self.adaptive_loss = bool(conf[8] == "True")
    # self.init_batch_size = int(conf[9])
    # self.batch_size = int(conf[9])
    # self.dropout_rate = float(conf[10])



    self.hard_resize = None
    self.no_pad_resize = False

    #self.pretain_path = os.path.join(_HERE(),"pretrained_models/res18")
    self.pretrain_path = os.path.join(_HERE(), "pretrained_models/eff/",self.pretraining, self.network_code)

    #tf.reset_default_graph()
    [t.join() for t in threads]
    self.tf_session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    # Show system inf o
    logger.info("System info ('uname -a'):")
    os.system('uname -a')
    # Show available devices
    local_device_protos = device_lib.list_local_devices()
    logger.info("Available local devices:\n{}".format(local_device_protos))
    # Show CUDA version
    logger.info("CUDA version:")
    os.system('nvcc --version')
    logger.info("Output of the command line 'nvclock':")
    os.system('nvclock')
    self.iterator_feed_handle, self.labels, self.logits, self.predictions, self.loss, self.accuracy, self.recall, self.trainOp, self.is_training, self.global_step = [None]*10

    load_method = "None"

    # does not work !!!
    # if load_method == "process":
    #   from multiprocessing import Process, Queue
    #   self.qu = Queue()
    #   self.preproc_thread = Process(target=init_model, args=(self.tf_session, metadata, self.pretrain_path, self.fc_size, self.wd, self.lr, self.network_code,self.dropout_rate,self.fc_bottleneck, self.qu))
    #   self.preproc_thread.daemon = True
    #   self.preproc_thread.start()
    #
    # elif load_method == "thread":
    #   import queue
    #   self.qu = queue.Queue()
    #   self.preproc_thread = threading.Thread(target=init_model(self.tf_session, metadata, self.pretrain_path, self.fc_size, self.wd, self.lr, self.network_code,self.dropout_rate, self.fc_bottleneck, self.qu))
    #   self.preproc_thread.daemon = True
    #   self.preproc_thread.start()
    #
    # else:
    import queue
    self.qu = queue.Queue()
    self.preproc_thread = None
    init_model(self.tf_session, metadata, self.pretrain_path, self.fc_size, self.network_code, self.dropout_rate, self.fc_bottleneck, self.qu)

  def np_one_hot(self, preds):
    b = np.zeros((preds.shape[0], self.number_of_labels))
    b[np.arange(preds.shape[0]), preds.astype(dtype=np.int32)] = 1
    return b



  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
    print(remaining_time_budget)

    if self.train_iterator_handle is None and self.train_data is None:


      # Reset TF graph
      dataset, self.mean_sizes, self.no_pad_resize, self.isMultilabel = prepare_dataset(dataset,self.do_multilabel_check,self.metadata,self.min_edge,self.max_edge,
                                self.default_factor,self.input_shape,is_test=False,resize_shape=self.hard_resize, resize_no_pad=self.no_pad_resize)


      # initialize iterators
      dataset_normal = dataset.batch(self.batch_size).prefetch(self.batch_size*2)
      dataset_normal = dataset_normal.repeat(-1)
      self.train_iterator = dataset_normal.make_one_shot_iterator()
      self.train_iterator_handle = self.tf_session.run(self.train_iterator.string_handle())

      dataset_init = dataset.batch(self.init_batch_size).prefetch(self.init_batch_size*2)
      dataset_init = dataset_init.repeat(-1)
      self.train_iterator_init = dataset_init.make_one_shot_iterator()
      self.train_iterator_handle_init = self.tf_session.run(self.train_iterator_init.string_handle())


      #self.iterator_feed_handle, self.labels, self.logits, self.predictions, self.loss, self.accuracy, self.recall, self.trainOp, self.is_training = None
      if self.preproc_thread is not None:
        self.preproc_thread.join()
      return_elements = []
      for x in [self.iterator_feed_handle, self.labels, self.logits, self.predictions, self.accuracy, self.recall,  self.is_training]:
        return_elements.append(self.qu.get())

      [self.iterator_feed_handle, self.labels, self.logits, self.predictions, self.accuracy, self.recall, self.is_training] = return_elements

      if self.isMultilabel is not None and self.isMultilabel == False and self.adaptive_loss == True:
        print("*************************")
        print("using softmax - crossent")
        self.loss = softmas_cross_entropy_loss_compute(self.labels, self.logits)
      else:
        print("*************************")
        print("using multilabel loss")
        self.loss = sigmoid_cross_entropy_with_logits(self.labels, self.logits)
      self.trainOp, self.global_step = TrainOp_EffiNet(self.loss, self.wd, self.lr)
      #self.trainOp = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.trainOp)
      # Weights initialization:
      init_new_vars_op = tf.variables_initializer(tf.global_variables())
      self.tf_session.run(init_new_vars_op)

    # Do Training
    sample_count = 0
    patience_count = 0

    while sample_count < int(20+20*self.train_count): # when do we return?
      try:

        if self.train_count < 5:
          # use tf.data iterator
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp],
                                                                         feed_dict={self.iterator_feed_handle: self.train_iterator_handle_init, self.is_training: True})
        else:
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp],
                                                                           feed_dict={self.iterator_feed_handle: self.train_iterator_handle, self.is_training: True})

        _predictions = np.argmax(_predictions,-1)
        sample_count += 1
        #print(_loss)
        if _loss < self.best_loss:
          self.best_loss = _loss
          patience_count = 0
        else:
          patience_count += 1

        try:
          all_labels = np.concatenate((all_labels), _labels, axis=0)
          all_predictions = np.concatenate((all_predictions, _predictions), axis=0)
        except:
          all_labels = _labels
          all_predictions = _predictions
        if sample_count % 20 == 0:
          if len(all_predictions.shape) < 2:
              all_predictions = self.np_one_hot(all_predictions)                   
          try:
              _BAC = roc_auc_score(all_labels, all_predictions)
          except ValueError:
              _BAC = 0
          _accuracy = accuracy_score(all_labels, all_predictions)
          all_labels = 0
          all_predictions = 0
          logger.info("Training step: {:05d}, accuracy: {:0.5f}, AUC: {:0.5f}, loss: {:0.5f}.".format(sample_count, _accuracy, _BAC, _loss))
        if patience_count > self.learning_rate_patience:
          print("increase global step")
          self.tf_session.run(self.global_step.assign(self.tf_session.run(self.global_step) + 1))
          patience_count = 0
          self.learning_rate_patience = self.learning_rate_patience+50


      except tf.errors.OutOfRangeError:
        print("tf out of range Error")
        break
    logger.info("Number of training examples: {}".format(sample_count))

    self.train_count +=1
    self.done_training = False

    if (1200 - remaining_time_budget)> 300:
      self.done_training = True

  def test(self, dataset, remaining_time_budget=None):
    """Make predictions on the test set `dataset` (which is different from that
    of the method `train`).

    Args:
      Same as that of `train` method, except that the `labels` will be empty
          since this time `dataset` is a test set.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
    if self.test_iterator_handle is None and self.test_data is None:
      if self.hard_resize is not None:
        tar_size = self.hard_resize
      else:
        if min(self.metadata.get_tensor_size()) < 1:
          tar_size = self.mean_sizes
        else:
          tar_size = np.array(self.metadata.get_tensor_size()[0:2][::-1])
      dataset, _ ,_,_ = prepare_dataset(dataset, self.do_multilabel_check, self.metadata, self.min_edge, self.max_edge, self.default_factor, self.input_shape,
                                is_test=True, resize_shape=tar_size, resize_no_pad=self.no_pad_resize, mean_sizes=self.mean_sizes)



      dataset = dataset.batch(self.batch_size).prefetch(self.batch_size*20)
      self.test_iterator = dataset.make_initializable_iterator()
      self.test_iterator_handle = self.tf_session.run(self.test_iterator.string_handle())


    self.tf_session.run(self.test_iterator.initializer)

    # Restoring the checkpoint and start the evaluation:
    all_predictions = np.zeros([1, self.metadata.get_output_size()])
    sample_count = 0
    while True:
      try:
        _predictions = self.tf_session.run(self.predictions, feed_dict={self.iterator_feed_handle: self.test_iterator_handle, self.is_training: False})
        if len(_predictions.shape) < 2:
          all_predictions =np.concatenate((all_predictions, self.np_one_hot(_predictions)), axis=0)
        else: 
          all_predictions =np.concatenate((all_predictions, _predictions), axis=0)
        sample_count += 1
        #print("sample_count "+ str(sample_count))
      except tf.errors.OutOfRangeError:
        break
    logger.info("Number of test examples: {}".format(sample_count))
    predictions = all_predictions[1:, :]
    #predictions = np.ones(predictions.shape)

    # if self.isMultilabel==False:
    #   predictions = self.np_one_hot(np.argmax(predictions, 1))

    #self.saver.save(self.tf_session, os.path.join(self.pretain_path+"/continued", "model.ckpt"))

    self.FirstIteration = False
    return predictions

def augment(images):
    images = tf.expand_dims(images, 0)
    #images = color(images)
    #scale = tf.squeeze(tf.random_uniform([1], 1.1, 1.5))
    #input_size = (self.input_size, self.input_size, 3)
    #images = tf.image.resize_nearest_neighbor(images, [tf.cast(tf.multiply(float(input_size[0]), scale), tf.int32), tf.cast(tf.multiply(float(input_size[0]), scale), tf.int32)])
    #random_angles = tf.random.uniform([1], -np.pi/4, np.pi/4)
    #images = tf.contrib.image.rotate(images, random_angles)
    #images = tf.random_crop(images, [1, input_size[0], input_size[1], 3])
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.squeeze(images, axis=0)
    return images

def prepare_dataset(dataset, do_multilabel_check, metadata, min_edge, max_edge, default_factor, input_shape,
                    is_test=False, resize_shape=None, resize_no_pad=False, mean_sizes = None):
  isMultilabel = None
  if not is_test and (do_multilabel_check or min(metadata.get_tensor_size()) < 1):
    # get mean size:

    #dataset = dataset.batch(1).prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    examples_input = list()
    examples_labels = list()

    with tf.Session() as sess:
      try:
        for i in range(25):
          # load full dataset
          tensor_4d, y = sess.run(next_element)
          examples_input.append(np.array(tensor_4d).astype(dtype=np.float16))
          examples_labels.append(y)
      except:
        print("full dataset loaded at init")
    # detect task and estimate class imbalance

    # if min(self.metadata.get_tensor_size()) > 1:
    #     examples_labels = np.concatenate(examples_labels)
    #     examples_single = list()
    #     for inp in examples_input:
    #         inp = np.squeeze(inp)
    #         examples_single.extend(np.split(inp,inp.shape[0], axis=0))
    #     examples_input = examples_single
    # else:
    examples_labels = np.stack(examples_labels)

    if examples_labels.sum() == examples_labels.shape[0]:
      isMultilabel = False
    else:
      isMultilabel = True

    logger.info("*********** Detected multilabel:  " + str(isMultilabel) + " ***********************")
    print(tensor_4d.shape)
    # Need to load a lot of data for this!
    # est_class_distribution = examples_labels.sum(axis=0) / examples_labels.sum()
    # balance_weights = (1 / est_class_distribution)
    # balance_weights = balance_weights / np.sum(balance_weights) * self.metadata.get_output_size()  # let the weights spread around 1
    # self.balance_weights = balance_weights
    #
    # logger.info("Estimated class distribution:  " + str(est_class_distribution))
    shapes = np.array([x.shape for x in examples_input])
    mean_sizes = np.average(shapes, 0)[1:3]
    mean_sizes = mean_sizes.astype(np.int)
    # transpose to agree with tensorflow definition
    # self.mean_sizes  = self.mean_sizes[::-1]

    logger.info("Estimated mean size:  " + str(mean_sizes))

  # resize_shape set? --> resize to that shap  if resize_shape is None:
    if min(metadata.get_tensor_size()) < 1:
      resize_shape = mean_sizes
    else:
      resize_shape = np.array(metadata.get_tensor_size()[0:2][::-1])

  if min(resize_shape) > min_edge/default_factor:
    resize_shape = (resize_shape*default_factor).astype(np.int)
  if max(resize_shape) > max_edge:
      resize_shape = (resize_shape/(max(resize_shape)/max_edge)).astype(np.int)
      if min(resize_shape) < min_edge:
        resize_shape[resize_shape == min(resize_shape)] = min_edge
        resize_no_pad = True

  # check if resize is necessary:
  if np.sum(metadata.get_tensor_size()[0:2] == resize_shape) !=2:
    input_shape = tuple(resize_shape)+(input_shape[-1],)
    logger.info("resizing to:  " + str(resize_shape))

    if resize_no_pad:
      # hard-resize to given shape
      TfResize = lambda x: tf.image.resize_images(x, resize_shape,method=tf.image.ResizeMethod.BILINEAR)
      def mapFunction_resize(x, y):
        return tf.map_fn(TfResize, x), y
      dataset = dataset.map(mapFunction_resize)

    else:

      # resize to given shape such that aspect ratio is preserved and both edges fit shape
      # pad the rest
      TfResize = lambda x : tf.image.resize_image_with_pad(
        #tf.expand_dims(x, 0),
        x,
        resize_shape[0],
        resize_shape[1],
        method=tf.image.ResizeMethod.BILINEAR
      )

      def mapFunction_resize(x, y):
        return tf.map_fn(TfResize, x), y
      dataset = dataset.map(mapFunction_resize)

  if not is_test:
   augment_l = lambda x: augment(x)
   def mapFunction(x, y):
     return tf.map_fn(augment_l, x), y

   dataset = dataset.map(mapFunction)

  return dataset, mean_sizes, resize_no_pad, isMultilabel

def init_model(tf_session, metadata, pretain_path, fc_size,network_code,dropout_rate,fc_bottleneck, qu):
  # init the model, metrics and training operator:
  # handle based iterator
  iterator_feed_handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
    iterator_feed_handle, (tf.float32, tf.float32))

  element, labels = iterator.get_next()
  labels = tf.cast(labels, dtype=tf.float32)
  element = tf.squeeze(element, axis=1)

  # set shape of iterator element such that dense works
  element.set_shape((None, None, None, metadata.get_tensor_size()[-1]))

  # build model
  logits, predictions, is_training = model(element, metadata, fc_size, network_code, dropout_rate, fc_bottleneck)

  logits = logits
  # if self.isMultilabel:
  #   logger.info("Loss function: Mutli-label loss function!")
  #   self.loss = self.multilabel_loss_compute(self.labels, logits)
  # else:
  #   logger.info("Loss function: Softmax Cross Entropy!")
  #   self.loss = self.multiclass_loss_compute(self.labels, logits, self.balance_weights)

  # loss = sigmoid_cross_entropy_with_logits(labels, logits)

  # self.AddRegularizationLoss(regularization)
  predictions = tf.cast(predictions, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))
  recall = tf.divide(tf.reduce_sum(tf.multiply(tf.cast(tf.equal(labels, predictions), dtype=tf.float32), labels)), tf.reduce_sum(labels))

  # self.trainOp = self.TrainOp(self.loss, learning_rate=self.learning_rate)
  # trainOp, global_step = TrainOp_EffiNet(loss, wd, lr)

  logger.info("Shape of dataset: {}".format(metadata.get_tensor_size()))
  #logger.info("Number of input: {}".format(input_shape))
  logger.info("Number of classes: {}".format(metadata.get_output_size()))



  ##############################
  #   pretrained model(s) here #
  ##############################
  # Find the latest checkpoint:
  latest_checkpoint = tf.train.latest_checkpoint(pretain_path)

  # Load pretrained
  all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  all_variables = [x for x in all_variables
                   if "Adam" not in x.name
                   and "FullyConnected" not in x.name
                   and "Preprocessing" not in x.name
                   and "_power" not in x.name
                   and "global_step_var" not in x.name
                   and "head" not in x.name]
  saver1 = tf.train.Saver(var_list=all_variables)
  if latest_checkpoint is not None:
    saver1.restore(tf_session, latest_checkpoint)
  else:
    print("#########################################")
    print("#### no pretrained weights found !!! ####")
    print("#########################################")

  for x in [iterator_feed_handle, labels, logits, predictions, accuracy, recall, is_training]:
    qu.put(x)

  return

def base_model(images, metadata, fc_size, network_code,dropout_rate, fc_bottleneck = True):

  is_training = tf.placeholder(tf.bool)
  # create the base pre-trained model

  net = images
  if metadata.get_tensor_size()[-1] != 3:
    with tf.variable_scope("Preprocessing"):
      net = tf.layers.conv2d(net, filters=3, kernel_size=[3, 3], padding="SAME")

  # ** Efficient Net **
  base_model = EfficientNet(model_name=network_code, num_classes=256)

  # use no specific scope
  net, endpoints = base_model.model(net, True)
  net = endpoints['global_pool']
  # net = tf.layers.flatten(net)
  net = tf.reduce_mean(net, [1, 2])

  if fc_bottleneck:
    with tf.variable_scope("FullyConnected_base"):
      # if self.isMultilabel:
      #   net = tf.contrib.layers.fully_connected(net, num_outputs=self.number_of_labels*2)
      #   #net = tf.layers.dropout(net, rate=self.dropout)
      #   net = tf.reshape(net, [-1, self.number_of_labels, 2])
      # else:
      #   net = tf.contrib.layers.fully_connected(net, num_outputs=self.number_of_labels)
      #   #net = tf.layers.dropout(net, rate=self.dropout)
      # net = tf.layers.dropout(
      #   inputs=net, rate=0.3,
      #   training=self.is_training)
      net = tf.layers.dense(inputs=net, units=fc_size, activation=tf.nn.relu)
  net = tf.layers.dropout(
    inputs=net, rate=dropout_rate,
    training=is_training)

  return net, is_training

def model(images, metadata, fc_size, network_code, dropout_rate, fc_bottleneck):

  number_of_labels = metadata.get_output_size()

  net, is_training = base_model(images, metadata, fc_size, network_code, dropout_rate, fc_bottleneck)

  with tf.variable_scope("FullyConnected_final"):
    net = tf.layers.dense(inputs=net, units=number_of_labels)

  # **Resnet-18**
  # base_model = ResNet(tf.train.get_global_step(), self.is_training, self.number_of_labels)
  # net = base_model.build_tower(net)

  logits = net
  probabilities = tf.nn.sigmoid(logits, name="sigmoid_tensor")

  return logits, probabilities, is_training

def multilabel_loss_compute(labels, logits):
  one_hot_labels = tf.one_hot(indices=tf.cast(labels, dtype=tf.uint8), depth=2)
  logits = tf.nn.softmax(logits, axis=-1) + 1e-6
  loss0 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1), 1 - labels)), 1e-6 + tf.reduce_sum(1 - labels))
  loss1 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1), labels)), 1e-6 + tf.reduce_sum(labels))
  return tf.add(loss0, loss1)

def softmas_cross_entropy_loss_compute(labels, logits, weights=None):
  if weights is None:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
  else:
    weighted_logits = tf.multiply(weights, logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=weighted_logits))
  return loss

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits

  return tf.reduce_sum(element_wise_xent)

def AddRegularizationLoss(regularization, loss):
  trainable_variables = tf.global_variables()
  reg_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
  for item in trainable_variables:
    if not "bias" in item.name:
      reg_loss = tf.add(reg_loss, tf.nn.l2_loss(item))
  return tf.add(loss, regularization * reg_loss)

# def TrainOp(self, loss, scope=None, learning_rate=0.256, momentum=0.9, optimizer=tf.train.MomentumOptimizer):
#   optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-3)
#   var_list = tf.global_variables()  # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
#   self.reg_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
#   for item in var_list:
#     self.reg_loss = tf.add(self.reg_loss, tf.nn.l2_loss(item))
#   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
#   with tf.control_dependencies(update_ops):
#     # with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
#     gvs = optimizer.compute_gradients(loss)
#     capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#     train_op = optimizer.apply_gradients(capped_gvs)
#     # train_op = optimizer.minimize(loss, var_list=var_list)
#   with tf.control_dependencies([train_op]):
#     optimizer2 = tf.train.MomentumOptimizer(learning_rate=1e-5 / 2, momentum=0)
#     train_op2 = optimizer2.minimize(self.reg_loss, var_list=var_list)
#   return tf.group(train_op, train_op2)

def simple_TrainOp(loss):
  optimizer = tf.train.AdamOptimizer(0.001)
  train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())
  return train_op

def TrainOp_EffiNet(loss, wd, lr):
  global_step = tf.Variable(0, trainable=False, name="global_step_var")
  learning_rate_decay = tf.compat.v1.train.exponential_decay(lr,
                                                       global_step,
                                                       decay_steps = 1,
                                                       decay_rate = 0.8,
                                                       staircase=True)

  optimizer = tf.contrib.opt.AdamWOptimizer(
    weight_decay=wd,
    learning_rate=learning_rate_decay,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='AdamW'
  )
  train_op = optimizer.minimize(
    loss=loss)
  return train_op, global_step




  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')

def prepare_meta_train(train_ds,test_ds,net,labels,wd,lr,tf_session, min_edge, max_edge, default_factor, is_training, nr =0):
  prepared_dict = {}
  # create last fc
  number_of_labels = train_ds.get_metadata().get_output_size()
  with tf.variable_scope("FullyConnected_final_"+str(nr)):
    net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
    net = tf.layers.dropout(
      inputs=net, rate=0.5,
      training=is_training)
    net = tf.layers.dense(inputs=net, units=number_of_labels)

  logits = net
  probabilities = tf.nn.sigmoid(logits, name="sigmoid_tensor")

  # create loss & train ops
  loss = sigmoid_cross_entropy_with_logits(labels, logits)

  predictions = tf.cast(probabilities, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))
  recall = tf.divide(tf.reduce_sum(tf.multiply(tf.cast(tf.equal(labels, predictions), dtype=tf.float32), labels)), tf.reduce_sum(labels))

  # self.trainOp = self.TrainOp(self.loss, learning_rate=self.learning_rate)
  trainOp,_ = TrainOp_EffiNet(loss, wd, lr) #forget about global step

  # Weights initialization:
  init_new_vars_op = tf.variables_initializer(tf.global_variables())
  tf_session.run(init_new_vars_op)

  prepared_dict["logits"] = logits
  prepared_dict["loss"] = loss
  prepared_dict["predictions"] = predictions
  prepared_dict["accuracy"] = accuracy
  prepared_dict["recall"] = recall
  prepared_dict["trainOp"] = trainOp

  # prepare ds
  prepped_train, mean_sizes = prepare_dataset(train_ds.get_dataset(), False, train_ds.get_metadata(), min_edge, max_edge, default_factor, input_shape = train_ds.get_metadata().get_tensor_size(),
                            is_test=False, resize_shape=None, resize_no_pad=False, mean_sizes=None)

  # initialize iterators
  prepped_train = prepped_train.batch(64).prefetch(1)
  prepped_train = prepped_train.repeat(-1)
  train_iterator = prepped_train.make_one_shot_iterator()
  train_iterator_handle = tf_session.run(train_iterator.string_handle())
  prepared_dict["train_iterator"] = train_iterator
  prepared_dict["train_iterator_handle"] = train_iterator_handle

  prepped_test, mean_sizes = prepare_dataset(test_ds.get_dataset(), False, train_ds.get_metadata(), min_edge, max_edge, default_factor, input_shape = train_ds.get_metadata().get_tensor_size(),
                            is_test=False, resize_shape=None, resize_no_pad=False, mean_sizes=mean_sizes)

  prepped_test = prepped_test.batch(64).prefetch(1)
  test_iterator = prepped_test.make_initializable_iterator()
  test_iterator_handle = tf_session.run(test_iterator.string_handle())
  prepared_dict["test_iterator"] = test_iterator
  prepared_dict["test_iterator_handle"] = test_iterator_handle

  return prepared_dict

def get_dataset_handle(path, train_test = "train"):
  # basename = Hammer.data
  basename = [x for x in os.listdir(path) if ".data" in x and "zip" not in x][0]
  return AutoDLDataset(os.path.join(path, basename, train_test))


if __name__ == '__main__':
  print("*******************************")
  print("**** called model.py main *****")
  print("*******************************")
  print("****     meta training    *****")
  print("*******************************")

  training_time = 3600 * 48
  architecture = "efficientnet-b2xmini"
  pretrain_dir_name = "metapretrained_models"


  import time
  start = time.time()

  root_dir = _HERE()
  root_dir = os.path.join(root_dir,"autodl")
  dataset_dir = os.path.join(root_dir, "AutoDL_public_data")
  default_ingestion_program_dir = os.path.join(root_dir, "AutoDL_ingestion_program")

  pretrain_dir = os.path.join(_HERE(), pretrain_dir_name)
  try:
    os.mkdir(pretrain_dir)
  except:
    pass

  pretrain_dir = os.path.join(pretrain_dir, architecture)
  try:
    os.mkdir(pretrain_dir)
  except:
    pass

  sys.path.append(default_ingestion_program_dir)
  from dataset import AutoDLDataset
  folders = [x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, x))]

  #folders = folders[:1]

  # folders = [x for x in folders if single_ds in x]
  blacklist = ["Katze","Kraut","Kreatur","Munster","Chucky"]
  folders = [x for x in folders if x not in blacklist]
  # load all datasets
  ds_handles_train = []
  for fo in folders:
    ds_handles_train.append(get_dataset_handle(os.path.join(dataset_dir, fo)))

  ds_handles_test = []
  for fo in folders:
    ds_handles_test.append(get_dataset_handle(os.path.join(dataset_dir, fo), "test"))

  ds_solution = []
  for fo in folders:
    # look for a .solution file
    ds_solution.append(np.loadtxt(os.path.join(dataset_dir, fo,
                                               [x for x in os.listdir(os.path.join(dataset_dir, fo)) if ".solution" in x][0])).astype(np.uint8))

  # init model until last fc
  metadata = ds_handles_train[0].get_metadata()
  # init the model, metrics and training operator:
  # handle based iterator
  iterator_feed_handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
    iterator_feed_handle, (tf.float32, tf.float32))

  element, labels = iterator.get_next()
  labels = tf.cast(labels, dtype=tf.float32)
  element = tf.squeeze(element, axis=1)

  # set shape of iterator element such that dense works
  element.set_shape((None, None, None, metadata.get_tensor_size()[-1]))

  net, is_training = base_model(element, metadata, 256, architecture, fc_bottleneck = False)

  # maybe load weights
  tf_sess = tf.Session()
  latest_checkpoint = tf.train.latest_checkpoint(pretrain_dir)

  # Load pretrained
  all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  all_variables = [x for x in all_variables
                   if "Adam" not in x.name
                   and "FullyConnected_" not in x.name
                   and "Preprocessing" not in x.name
                   and "_power" not in x.name
                   and "head" not in x.name]
  saver = tf.train.Saver(var_list=all_variables, max_to_keep = 1)

  if latest_checkpoint is not None:
    saver.restore(tf_sess, latest_checkpoint)


  # create last fc and optimizer - per dataset

  prepared_ds = []
  best_acc = []
  for train_ds, test_ds, nr in zip(ds_handles_train,ds_handles_test, range(len(ds_handles_test))):
    prepared_ds.append(prepare_meta_train(train_ds, test_ds,net,labels, wd=0.000025,lr=0.0001, tf_session=tf_sess,
                                          min_edge=64, max_edge=256, default_factor=0.5, is_training= is_training, nr = nr))
    best_acc.append(-1E10)

  # train for a given amount of time
  while (time.time()-start)<training_time:

    # evaluate on test set
    accuracy = []
    for ds, id in zip(prepared_ds, range(len(prepared_ds))):

      tf_sess.run(ds["test_iterator"].initializer)

      predictions = []
      while True:
        try:
          _predictions = tf_sess.run([ds["predictions"]], feed_dict={iterator_feed_handle: ds["test_iterator_handle"],  is_training: False})
          predictions.append(_predictions[0])
        except tf.errors.OutOfRangeError:
          break
      # maybe save weights
      predictions = np.concatenate(predictions)
      true_mass = sum(predictions[ds_solution[id]==1])* np.sum(ds_solution[id]==0)/predictions.size
      false_mass = sum(predictions[ds_solution[id] == 0]) * np.sum(ds_solution[id] == 1) / predictions.size
      accuracy.append(true_mass-false_mass)

    improvement = 0
    # relative changes are positive
    if np.sum((np.array(accuracy) - np.array(best_acc)) / np.abs(np.array(best_acc))) > 0:
      best_acc = accuracy
      print(str(time.localtime())+"new best accuracy:")
      print(best_acc)
      saver.save(tf_sess, pretrain_dir+"/model")
    else:
      print("no new best accuracy, new:")
      print(accuracy)


    # train for a number of batches per dataset
    for i in range(200):
      for ds in prepared_ds:
        _loss, _labels, _predictions, _logits, _ = tf_sess.run([ds["loss"],labels,ds["predictions"],ds["logits"],ds["trainOp"]],
                                                                       feed_dict={iterator_feed_handle: ds["train_iterator_handle"],  is_training: True})





