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
import tensorflow as tf
sys.path.append("./python_packages")
sys.path.append(_HERE()+"/efficientnet")
sys.path.append(_HERE()+"/autoaugment")
sys.path.append(_HERE()+"/res18")
from efficientnet import EfficientNet
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from resnet import ResNet
from PIL import Image

class Model(object):
  """Trivial example of valid model. Returns all-zero predictions."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
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

    self.FirstIteration = True
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = 64
    self.max_edge = 128

    self.lr = 0.001
    self.wd = 1e-5

    text_file = open("config.txt", "r")
    conf = text_file.read()
    text_file.close()
    conf = conf.split("_")
    self.lr = float(conf[0])
    self.wd = float(conf[1])

    self.hard_resize = None
    self.no_pad_resize = False
    self.py_data_processing = False
    #self.pretain_path = os.path.join(_HERE(),"pretrained_models/res18")
    self.pretain_path = os.path.join(_HERE(), "pretrained_models/eff/efficientnet-b0")

    #tf.reset_default_graph()
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
    
  def np_one_hot(self, preds):
    b = np.zeros((preds.shape[0], self.number_of_labels))
    b[np.arange(preds.shape[0]), preds.astype(dtype=np.int32)] = 1
    return b
    
  def color(self, x):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
    


  def model(self, images, keep_prob=1.0, number_of_classes=2, is_training=True):

    self.number_of_labels = self.metadata.get_output_size()
    self.dropout = tf.placeholder(tf.float32)
    self.is_training = tf.placeholder(tf.bool)
    # create the base pre-trained model

    net = images
    if self.metadata.get_tensor_size()[-1] != 3:
      with tf.variable_scope("Preprocessing"):
        net = tf.layers.conv2d(net, filters=3, kernel_size=[3, 3], padding="SAME")


    #** Efficient Net **
    base_model = EfficientNet(model_name="efficientnet-b0", num_classes=256)

    #use no specific scope
    net, endpoints = base_model.model(net, True)
    net = endpoints["block_15/expansion_output"]
    #net = tf.layers.flatten(net)
    net = tf.reduce_mean(net, [1, 2])

    with tf.variable_scope("FullyConnected"):
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
      net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
      net = tf.layers.dropout(
        inputs=net, rate=0.5,
        training=self.is_training)
      net = tf.layers.dense(inputs=net, units=self.number_of_labels)

    # **Resnet-18**
    # base_model = ResNet(tf.train.get_global_step(), self.is_training, self.number_of_labels)
    # net = base_model.build_tower(net)



          
    logits = net
    probabilities = tf.nn.sigmoid(logits, name="sigmoid_tensor")

    return logits, probabilities


  def multilabel_loss_compute(self, labels, logits):
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, dtype=tf.uint8), depth=2)
    logits = tf.nn.softmax(logits, axis=-1)+1e-6
    loss0 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1), 1-labels)), 1e-6+tf.reduce_sum(1-labels))
    loss1 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1),   labels)), 1e-6+tf.reduce_sum(  labels))  
    return tf.add(loss0, loss1)


  def multiclass_loss_compute(self,labels,logits, weights=None):
    if weights is None:
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    else:
      weighted_logits = tf.multiply(weights, logits)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=weighted_logits))
    return loss

  def sigmoid_cross_entropy_with_logits(self, labels=None, logits=None):
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


  def AddRegularizationLoss(self, regularization):
    self.trainable_variables = tf.global_variables()
    self.reg_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
    for item in self.trainable_variables:
      if not "bias" in item.name:
        self.reg_loss = tf.add(self.reg_loss, tf.nn.l2_loss(item))
    return tf.add(self.loss, regularization*self.reg_loss)


  def TrainOp(self, loss, scope=None, learning_rate=0.256, momentum=0.9, optimizer=tf.train.MomentumOptimizer):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-3)
    var_list = tf.global_variables()  # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    self.reg_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
    for item in var_list:
      self.reg_loss = tf.add(self.reg_loss, tf.nn.l2_loss(item))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    with tf.control_dependencies(update_ops):
      # with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
      gvs = optimizer.compute_gradients(loss)
      capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      train_op = optimizer.apply_gradients(capped_gvs)
      # train_op = optimizer.minimize(loss, var_list=var_list)
    with tf.control_dependencies([train_op]):
      optimizer2 = tf.train.MomentumOptimizer(learning_rate=1e-5 / 2, momentum=0)
      train_op2 = optimizer2.minimize(self.reg_loss, var_list=var_list)
    return tf.group(train_op, train_op2)

  def simple_TrainOp(self, loss):
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
    return train_op


  def TrainOp_EffiNet(self, loss):
    optimizer = tf.contrib.opt.AdamWOptimizer(
    weight_decay=self.wd,
    learning_rate=self.lr,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='AdamW'
    )
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
    return train_op


  def prepare_dataset(self, dataset, is_test=False, resize_shape=None, resize_no_pad=False, py_data_processing=False):

    if not is_test:
      # get mean size:
      if min(self.metadata.get_tensor_size()) > 1:
        dataset = dataset.batch(200).prefetch(1)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      examples_input = list()
      examples_labels = list()
      with tf.Session() as sess:
        try:
          while True:  # load full dataset
            tensor_4d, y = sess.run(next_element)
            examples_input.append(np.array(tensor_4d).astype(dtype=np.float16))
            examples_labels.append(y)
        except:
          print("full dataset loaded at init")
      # detect task and estimate class imbalance

      if min(self.metadata.get_tensor_size()) > 1:
          examples_labels = np.concatenate(examples_labels)
          examples_single = list()
          for inp in examples_input:
              inp = np.squeeze(inp)
              examples_single.extend(np.split(inp,inp.shape[0], axis=0))
          examples_input = examples_single
      else:
          examples_labels = np.stack(examples_labels)

      if examples_labels.sum() == examples_labels.shape[0]:
        self.isMultilabel = False
      else:
        self.isMultilabel = True

      logger.info("*********** Detected multilabel:  " + str(self.isMultilabel) + " ***********************")

      est_class_distribution = examples_labels.sum(axis=0) / examples_labels.sum()
      balance_weights = (1 / est_class_distribution)
      balance_weights = balance_weights / np.sum(balance_weights) * self.metadata.get_output_size()  # let the weights spread around 1
      self.balance_weights = balance_weights

      logger.info("Estimated class distribution:  " + str(est_class_distribution))
      shapes = np.array([x.shape for x in examples_input])
      self.mean_sizes = np.average(shapes, 0)[1:3]

      # transpose to agree with tensorflow definition
      self.mean_sizes  = self.mean_sizes[::-1]

      logger.info("Estimated mean size:  " + str(self.mean_sizes))

    if is_test and py_data_processing:
      # get mean size:
      if min(self.metadata.get_tensor_size()) > 1:
        dataset = dataset.batch(200).prefetch(1)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      examples_input = list()
      examples_labels = list()
      with tf.Session() as sess:
        try:
          while True:  # load full dataset
            tensor_4d, y = sess.run(next_element)
            examples_input.append(np.array(tensor_4d).astype(dtype=np.float16))
            examples_labels.append(y)
        except:
          print("full dataset loaded at init")
      # detect task and estimate class imbalance


      if min(self.metadata.get_tensor_size()) > 1:
          examples_labels = np.concatenate(examples_labels)
          examples_single = list()
          for inp in examples_input:
              inp = np.squeeze(inp)
              examples_single.extend(np.split(inp, inp.shape[0], axis=0))
          examples_input = examples_single
      else:
          examples_labels = np.stack(examples_labels)

    # resize_shape set? --> resize to that shape
    if resize_shape is None:
      resize_shape = self.mean_sizes
    if max(resize_shape)>self.max_edge:
      resize_shape = (self.mean_sizes/(max(self.mean_sizes)/self.max_edge)).astype(np.int)


    # check if resize is necessary:
    if np.sum(self.metadata.get_tensor_size()[0:2] == resize_shape) !=2:
      self.input_shape = tuple(resize_shape)+(self.input_shape[-1],)
      logger.info("resizing to:  " + str(resize_shape))
      if py_data_processing == True:
        from PIL import Image, ImageOps

        if resize_no_pad:
          # hard-resize to given shape
          for ix, ele in enumerate(examples_input):
            img = Image.fromarray((ele[0]*256).astype(np.uint8))
            #img.save("origi.jpg")
            img = img.resize(resize_shape, Image.BILINEAR)
            #img.save("resizeovic.jpg")
            img = np.array(img)/256

            examples_input[ix] = img.transpose((1,0,2)).astype(dtype=np.float16)
        else:
          # resize to given shape such that aspect ratio is preserved and both edges fit shape
          # pad the rest
          for ix, ele in enumerate(examples_input):
            resize_coef = np.min(resize_shape / np.array(ele.shape[1:3])[::-1])
            img = Image.fromarray((ele[0]*256).astype(np.uint8))
            img = img.resize(tuple(np.floor(np.array(img.size)*resize_coef).astype(np.int)))
            offsets = (resize_shape - np.array(img.size))/2
            offsets_fl = np.floor(offsets).astype(np.int)
            offsets_ceil = np.ceil(offsets).astype(np.int)
            img = ImageOps.expand(img,(offsets_fl[0], offsets_fl[1], offsets_ceil[0], offsets_ceil[1]))
            img = np.array(img)/256

            examples_input[ix] = img.transpose((1,0,2)).astype(dtype=np.float16)
        print("return full dataset instead of handle")
        examples_input = np.stack(examples_input, 0)
        examples_labels = np.stack(examples_labels, 0)
        return [examples_input, examples_labels]
      else:
        if resize_no_pad:
          # hard-resize to given shape
          TfResize = lambda x: tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(x, 0), resize_shape), 0)
          def mapFunction_resize(x, y):
            return tf.map_fn(TfResize, x), y
          dataset = dataset.map(mapFunction_resize)

        else:

          # resize to given shape such that aspect ratio is preserved and both edges fit shape
          # pad the rest
          TfResize = lambda x : tf.squeeze(tf.image.resize_image_with_pad(
            #tf.expand_dims(x, 0),
            x,
            resize_shape[0],
            resize_shape[1],
            method=tf.image.ResizeMethod.BILINEAR
          ), 0)

          def mapFunction_resize(x, y):
            return tf.map_fn(TfResize, x), y
          dataset = dataset.map(mapFunction_resize)
    elif py_data_processing:
      print("return full dataset instead of handle")
      examples_input = np.concatenate(examples_input, 0)
      examples_labels = np.stack(examples_labels, 0)
      return [examples_input, examples_labels]
    return dataset

  def get_next_batch(self, is_train = True):
    if is_train:
      # check if epoch has been completed
      if self.train_batch_pointer+self.batch_size > self.train_data.shape[0]:
        batch_tr = self.train_data[self.train_batch_pointer:self.train_data.shape[0]]
        batch_lb = self.train_labels[self.train_batch_pointer:self.train_data.shape[0]]
        self.train_batch_pointer = 0
        return [batch_tr, batch_lb], True
      else:
        point_new = self.train_batch_pointer + self.batch_size
        batch_tr = self.train_data[self.train_batch_pointer:point_new]
        batch_lb = self.train_labels[self.train_batch_pointer:point_new]
        self.train_batch_pointer = point_new
        return [batch_tr, batch_lb], False

    else:
      if self.test_batch_pointer + self.batch_size > self.test_data.shape[0]:
        batch_tr = self.test_data[self.test_batch_pointer:self.test_data.shape[0]]
        self.test_batch_pointer = 0
        return [batch_tr], True
      else:
        point_new = self.test_batch_pointer + self.batch_size
        batch_tr = self.test_data[self.test_batch_pointer:point_new]
        self.test_batch_pointer = point_new
        return [batch_tr], False

  def auto_augment(self,batch):
    if batch[0].shape[-1] ==3:
      for ix in range(0,batch[0].shape[0]):
        img = Image.fromarray((batch[0][ix]*256).astype(np.uint8))
        #img.save("pre_transform.jpg")
        img = self.augment_policy(img)
        batch[0][ix] = (np.array(img)/256).astype(dtype=np.float16)
    # else:
    #    print("Augmentation only works for 3 input channels")
    return batch


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

    if self.train_iterator_handle is None and self.train_data is None:

      self.input_shape = self.metadata.get_tensor_size()
      # Reset TF graph
      dataset = self.prepare_dataset(dataset,is_test=False,resize_shape=self.hard_resize, resize_no_pad=self.no_pad_resize, py_data_processing=self.py_data_processing)

      if self.py_data_processing:
        print("set up feed placeholders")
        self.train_data, self.train_labels = dataset
        self.element = tf.placeholder(tf.float32, shape=[None,]+list(self.input_shape))
        self.labels = tf.placeholder(tf.float32, shape=[None, self.metadata.get_output_size()])

        self.train_batch_pointer = 0
        self.test_batch_pointer = 0

      else:
        # initialize iterators
        dataset = dataset.batch(self.batch_size).prefetch(1)
        dataset = dataset.repeat(-1)
        self.train_iterator = dataset.make_one_shot_iterator()
        self.train_iterator_handle = self.tf_session.run(self.train_iterator.string_handle())

        # init the model, metrics and training operator:
        # handle based iterator
        self.iterator_feed_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
          self.iterator_feed_handle, self.train_iterator.output_types)

        element, labels = iterator.get_next()
        self.labels = tf.cast(labels, dtype=tf.float32)
        self.element = tf.squeeze(element, axis=1)

        #set shape of iterator element such that dense works
        self.element.set_shape((None,) + self.input_shape)

      #build model
      logits, predictions = self.model(self.element)

      self.logits = logits
      # if self.isMultilabel:
      #   logger.info("Loss function: Mutli-label loss function!")
      #   self.loss = self.multilabel_loss_compute(self.labels, logits)
      # else:
      #   logger.info("Loss function: Softmax Cross Entropy!")
      #   self.loss = self.multiclass_loss_compute(self.labels, logits, self.balance_weights)

      self.loss = self.sigmoid_cross_entropy_with_logits(self.labels, self.logits)

      #self.AddRegularizationLoss(regularization)
      self.predictions = tf.cast(predictions, dtype=tf.float32)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), dtype=tf.float32))
      self.recall = tf.divide(tf.reduce_sum(tf.multiply(tf.cast(tf.equal(self.labels, self.predictions), dtype=tf.float32), self.labels)), tf.reduce_sum(self.labels))
      
      #self.trainOp = self.TrainOp(self.loss, learning_rate=self.learning_rate)
      self.trainOp = self.TrainOp_EffiNet(self.loss)

      logger.info("Shape of dataset: {}".format(self.metadata.get_tensor_size()))
      logger.info("Number of input: {}".format(self.input_shape))
      logger.info("Number of classes: {}".format(self.metadata.get_output_size()))

      # Weights initialization:
      init_new_vars_op = tf.variables_initializer(tf.global_variables())
      self.tf_session.run(init_new_vars_op)

      ##############################
      #   pretrained model(s) here #
      ##############################
      # Find the latest checkpoint:
      latest_checkpoint = tf.train.latest_checkpoint(self.pretain_path)

      # Load pretrained
      all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      all_variables = [x for x in all_variables
                       if "Adam" not in x.name
                       and "FullyConnected" not in x.name
                       and "Preprocessing" not in x.name
                       and "_power" not in x.name
                       and "head" not in x.name]
      saver1 = tf.train.Saver(var_list=all_variables)
      saver1.restore(self.tf_session, latest_checkpoint)
    
      # Create the Saveru:
      # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # #saver = tf.train.Saver(var_list=var_list+update_ops)
      # saver = tf.train.Saver()
      # if latest_checkpoint is None:
      #   self.tf_session.run(init_new_vars_op)
      # else:
      #   try:
      #     saver.restore(self.tf_session, latest_checkpoint)
      #   except:
      #     logger.info("only restore scope Model")
      #     self.tf_session.run(init_new_vars_op)
      #     var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Model")
      #     update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Model")
      #     saver = tf.train.Saver(var_list=var_list2+update_ops2)
      #     saver.restore(self.tf_session, latest_checkpoint)
      #     saver = tf.train.Saver(var_list=var_list+update_ops)
      # self.saver = saver
    
    # Restoring the checkpoint or initialize the weights and start the training:
    sample_count = 0
    while sample_count < int(600): # when do we return?
      try:
        if self.py_data_processing:
          # feed images directly
          batch,_ = self.get_next_batch(is_train = True)
          if self.augment_policy is not None:
            batch = self.auto_augment(batch)
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp],
                                                                         feed_dict={self.element:batch[0], self.labels:batch[1], self.dropout: 0.5, self.learning_rate: 1e-6, self.is_training: True})

        else:
          # use tf.data iterator
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp],
                                                                         feed_dict={self.iterator_feed_handle: self.train_iterator_handle, self.dropout: 0.5, self.learning_rate: 1e-6, self.is_training: True})
        _predictions = np.argmax(_predictions,-1)
        sample_count += 1
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


      except tf.errors.OutOfRangeError:
        print("tf out of range Error")
        break
    logger.info("Number of training examples: {}".format(sample_count))

    self.done_training = False

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
        tar_size = self.mean_sizes
      dataset = self.prepare_dataset(dataset, is_test=True, resize_shape=tar_size, resize_no_pad=self.no_pad_resize, py_data_processing=self.py_data_processing)

      if self.py_data_processing:
        self.test_data = dataset[0]
      else:
        dataset = dataset.batch(self.batch_size).prefetch(1)
        self.test_iterator = dataset.make_initializable_iterator()
        self.test_iterator_handle = self.tf_session.run(self.test_iterator.string_handle())

    if not self.py_data_processing:
      self.tf_session.run(self.test_iterator.initializer)

    # Restoring the checkpoint and start the evaluation:
    all_predictions = np.zeros([1, self.metadata.get_output_size()])
    sample_count = 0
    while True:
      try:
        if self.py_data_processing:
          batch, epoch_done = self.get_next_batch(is_train=False)
          _predictions = self.tf_session.run(self.predictions, feed_dict={self.element: batch[0], self.dropout: 0, self.is_training: False})
        else:
          _predictions = self.tf_session.run(self.predictions, feed_dict={self.iterator_feed_handle: self.test_iterator_handle, self.dropout: 0, self.is_training: False})
        if len(_predictions.shape) < 2:
          all_predictions =np.concatenate((all_predictions, self.np_one_hot(_predictions)), axis=0)
        else: 
          all_predictions =np.concatenate((all_predictions, _predictions), axis=0)
        sample_count += 1
        #print("sample_count "+ str(sample_count))
        if self.py_data_processing and epoch_done:
          break
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