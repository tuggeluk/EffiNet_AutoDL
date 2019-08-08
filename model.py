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
import mobilenet_v2
import logging
import numpy as np
import time
import os


import tensorflow as tf
sys.path.append("./python_packages")
regularization = 1e-4

class Model(object):
  """Trivial example of valid model. Returns all-zero predictions."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_trai = metadata
    self.metadata = metadata
    self.train_iterator = None
    self.train_iterator_handle = None
    self.test_iterator = None
    self.test_iterator_handle = None
    self.FirstIteration = True
    self.is_training = tf.placeholder(tf.bool)
    self.output_dim = self.metadata.get_output_size()
    self.number_of_frames = 10
    self.learning_rate = 0.001
    self.seen_sample = 0
    self.batch_size = 25
    self.input_size = None
    self.time_budget = 1200
    self.start = time.time()
    self.test_time = 0
    self.training_time = 0
    self.done_training = False
    
    self.default_image_size = [112,112]
    self.image_size = metadata.get_matrix_size()
    if self.image_size[0] > 224:
        self.image_size = (224,224)
    if self.image_size[0] < 0 or self.image_size[1] < 0:
        self.image_size = self.default_image_size

    #tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    self.tf_session = tf.Session(config=config)

    # Show system info
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
    
  def Augment(self, images):
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
    
  def preprocess_tensor_4d(self, tensor):
    frames = self.number_of_frames
    input_shape = self.input_shape
    if self.input_shape[0]==-1 or self.input_shape[1]==-1:
        input_shape = self.default_image_size
        self.input_shape = self.default_image_size  
    tensor_shape = tensor.shape#  get_shape().as_list()
    '''
    if tensor_shape[0] < frames:
        tensor = tf.tile(tensor, [tf.cast(tf.math.floor(frames / tensor_shape[0]) + 1, tf.int32), 1, 1, 1])
        tensor = tensor[:frames, :, :, :]
    elif tensor_shape[0] > frames:
        tensor = tensor[0:tensor_shape[0]:int(tensor_shape[0]/frames), :, :, :]
        tensor = tensor[:frames, :, :, :]
    '''
    tensor = tf.cond(tf.logical_not(tf.logical_and(tf.is_nan(tf.cast(tf.shape(tensor)[0], tf.float32)), tf.cast(tf.shape(tensor)[0], tf.float32) < frames)), lambda: tf.tile(tensor, [tf.cast(tf.math.floor(frames / tf.cast(tf.shape(tensor)[0], tf.float32)) + 1, tf.int32), 1, 1, 1]), lambda: tensor) 
    tensor = tf.cond(tf.logical_not(tf.logical_and(tf.is_nan(tf.cast(tf.shape(tensor)[0], tf.float32)), tf.cast(tf.shape(tensor)[0], tf.float32) > frames)), lambda: tensor[0:tensor_shape[0]:tf.cast(tf.cast(tf.shape(tensor)[0], tf.float32)/frames, tf.int32), :, :, :], lambda: tensor)
    tensor = tensor[:frames, :, :, :]
    if True:#not input_shape is None:
        tensor = tf.image.resize_nearest_neighbor(tensor, input_shape)
    return tensor
    
    
  def model(self, features):
    input_layer = features

    # Replace missing values by 0
    hidden_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)
    

    if self.metadata.get_tensor_shape()[0] != -1:
        #hidden_layer = tf.squeeze(hidden_layer, axis=[1])
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
            logits, endpoints = mobilenet_v2.mobilenet(hidden_layer, self.input_size)
        hidden_layer = tf.contrib.layers.conv2d(
                inputs=endpoints['feature_maps'], num_outputs=1280, kernel_size=1, stride=1, activation_fn=None)
        hidden_layer = tf.reduce_mean(input_tensor=hidden_layer, axis=[1, 2])
    else:
        tensor_shape = hidden_layer.get_shape().as_list()
        tensor_reshape = tf.shape(hidden_layer)
        hidden_layer = tf.reshape(hidden_layer, [-1]+tensor_shape[-3:])
        if tensor_shape[-1] == 1:
            hidden_layer = tf.tile(hidden_layer, [1, 1, 1, 3])
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
            logits, endpoints = mobilenet_v2.mobilenet(hidden_layer, self.input_size)
            
        hidden_layer = endpoints['feature_maps']
        feature_shape = hidden_layer.get_shape().as_list()
        hidden_layer = tf.reshape(hidden_layer, [-1, self.number_of_frames]+feature_shape[-3:])
        hidden_layer = tf.reduce_mean(hidden_layer, axis=1)
        hidden_layer = tf.layers.flatten(hidden_layer)
          
    hidden_layer = tf.layers.dense(inputs=hidden_layer, units=256, activation=tf.nn.relu)
    hidden_layer = tf.layers.dropout(inputs=hidden_layer, rate=0.5, training=self.is_training)

    logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim)
    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")
    return logits, sigmoid_tensor
    
  def loss_compute(self, labels, logits):
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, dtype=tf.uint8), depth=2)
    logits = tf.nn.softmax(logits, axis=-1)+1e-6
    loss0 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1), 1-labels)), 1e-6+tf.reduce_sum(1-labels))
    loss1 = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(one_hot_labels, -tf.log(logits)), axis=-1),   labels)), 1e-6+tf.reduce_sum(  labels))  
    return tf.add(loss0, loss1)
    
  def sigmoid_cross_entropy_with_logits(self, labels=None, logits=None):
    labels = tf.cast(labels, dtype=tf.float32)
    relu_logits = tf.nn.relu(logits)
    exp_logits = tf.exp(- tf.abs(logits))
    sigmoid_logits = tf.log(1 + exp_logits)
    element_wise_xent = relu_logits - labels * logits + sigmoid_logits
    return tf.reduce_mean(tf.reduce_sum(element_wise_xent, 1))
    
  def AddRegularizationLoss(self, regularization):
    self.trainable_variables = tf.global_variables()
    self.reg_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
    for item in self.trainable_variables:
      if not "bias" in item.name:
        self.reg_loss = tf.add(self.reg_loss, tf.nn.l2_loss(item))
    return tf.add(self.loss, regularization*self.reg_loss)
  
    
  def TrainOp(self, loss, scope=None, learning_rate=1e-2, momentum=0.9, optimizer=tf.train.MomentumOptimizer):
    self.global_step = tf.Variable(0, trainable=False)
    optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0, learning_rate=self.learning_rate)
    train_op = optimizer.minimize(loss=loss)
    return train_op
  

  def train(self, dataset, remaining_time_budget=None):
    
    if self.train_iterator_handle is None:
      # Reset TF graph
      #dataset = dataset.batch(1).prefetch(1)
      if True:#self.metadata.get_tensor_shape()[1] == -1:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        
              
        examples_input = list()
        examples_labels = list()
        
        try:
          for i in range(25):
            # load full dataset
            tensor_4d, y = self.tf_session.run(next_element)
            examples_input.append(np.array(tensor_4d).astype(dtype=np.float16))
            examples_labels.append(y)
        except:
          print("full dataset loaded at init")
        examples_labels = np.stack(examples_labels)
        
        if examples_labels.sum() == examples_labels.shape[0]:
          self.isMultilabel = False
        else:
          self.isMultilabel = True
        
        logger.info("*********** Detected multilabel:  " + str(self.isMultilabel) + " ***********************")
        shapes = np.array([x.shape for x in examples_input])
        self.mean_sizes = np.average(shapes, 0)[1:3]
        shape = self.mean_sizes
        self.mean_sizes = self.mean_sizes.astype(np.int)
      else:
        shape = self.metadata.get_tensor_shape()[1:3]    
      
      print("Average shape:" ,shape) 
      if self.metadata.get_tensor_shape()[0] != -1:
        self.batch_size = 256          
        if np.max(shape) > 128:
          max_dim = (np.max(shape))/128
          self.input_shape = [int(shape[0]/max_dim), int(shape[1]/max_dim)]
          #self.input_shape = [224, 224]
          self.input_size = np.max(self.input_shape) 
        else:
          #self.input_shape = [224, 224]
          self.input_shape = shape
          self.input_size = np.max(self.input_shape)  
      else:
          self.input_shape = self.metadata.get_tensor_shape()[1:3]
          self.input_size = max(self.metadata.get_tensor_shape())     
        
      logger.info("*********** Input shape:  ***********************")
      print(self.input_shape)    
      if self.metadata.get_tensor_shape()[0] != -1:
        TfResize = lambda x: tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(x, 0), self.input_shape), 0)
        def mapFunction(x, y):
          return tf.map_fn(TfResize, x), y
        dataset = dataset.map(mapFunction)
        
        augment = lambda x: self.Augment(x)
        def mapFunction(x, y):
          return tf.map_fn(augment, x), y
        dataset = dataset.map(mapFunction) 
      else: 
        #dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))  
        #TfResize = lambda x: tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(x, 0), self.input_shape), 0)
        def mapFunction(x, y):     
          return tf.map_fn(self.preprocess_tensor_4d, tf.expand_dims(x, 0)), y
        dataset = dataset.map(mapFunction)  
      #if self.metadata.get_tensor_size()[0]==-1 or self.metadata.get_tensor_size()[0] > 32:
      #  self.batch_size = int(self.batch_size/4)
      print(self.batch_size)
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
      if self.metadata.get_tensor_shape()[0] != -1:
        element = tf.squeeze(element, axis=1)
      #set shape of iterator element such that dense works
      input_size = self.metadata.get_tensor_size()
      if self.metadata.get_tensor_shape()[0] != -1:
        element.set_shape((None,) + (self.input_shape[0], self.input_shape[1], input_size[-1]))
      else:
        element.set_shape((None, None,) + (self.input_shape[0], self.input_shape[1], input_size[-1]))

      logits, predictions = self.model(element)
      self.logits = logits
      if self.isMultilabel:
        logger.info("Loss function: Mutli-label loss function!")
        self.loss = self.sigmoid_cross_entropy_with_logits(labels, logits)
      else:
        logger.info("Loss function: Softmax Cross Entropy!")
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        self.loss = self.sigmoid_cross_entropy_with_logits(labels, logits)
      #self.AddRegularizationLoss(regularization)
      self.predictions = tf.cast(predictions, dtype=tf.float32)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, self.predictions), dtype=tf.float32))
      self.recall = tf.divide(tf.reduce_sum(tf.multiply(tf.cast(tf.equal(labels, self.predictions), dtype=tf.float32), labels)), tf.reduce_sum(labels))

      logger.info("Shape of example: {}".format(self.metadata.get_tensor_size()))
      logger.info("Number of classes: {}".format(self.metadata.get_output_size()))

      ######################################
      # TODO load pretrained model(s) here #
      ######################################
      # Find the latest checkpoint:
      latest_checkpoint = tf.train.latest_checkpoint("pretrained_models")
    
      # Create the Saver:
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      var_list = tf.global_variables()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      var_list_mobile = tf.global_variables(scope="MobilenetV2")[:-2]
    
      self.trainOp = self.TrainOp(self.loss, learning_rate=self.learning_rate)
      self.trainOp0 = self.trainOp#self.TrainOp(self.loss, scope="FullyConnected", learning_rate=self.learning_rate)
      self.tf_session.run(tf.initializers.global_variables())
      if self.FirstIteration:
        if latest_checkpoint is None:
          #self.tf_session.run(init_new_vars_op)
          print("Hi! :)")
          saver = tf.train.Saver()
        else:
          try:        
            saver2 = tf.train.Saver(var_list_mobile)
            saver2.restore(self.tf_session, latest_checkpoint)
            logger.info("Pretrained model is restored successfully!")
            #import time;time.sleep(3)
          except:
            logger.info("Failed to load Pretrained model!")
            #import time;time.sleep(3)
    
    # Restoring the checkpoint or initialize the weights and start the training:
    sample_count = 0
    if self.FirstIteration:
      self.training_steps = int(np.round(np.min([1e2+1, self.metadata.size()/self.batch_size+1])/2))
    else:
      self.training_steps *= 1.5
      self.training_steps = max(self.training_steps, 200)
    start_training = time.time()
    while sample_count < self.training_steps: # when do we return? TODO come up with smart heuristic
      try:
        if self.FirstIteration:                    
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp0], feed_dict={self.iterator_feed_handle: self.train_iterator_handle, self.is_training: True})
          self.FirstIteration = False
          sample_count += 1
        else:
          _loss, _labels, _predictions, _logits, _ = self.tf_session.run([self.loss, self.labels, self.predictions, self.logits, self.trainOp], feed_dict={self.iterator_feed_handle: self.train_iterator_handle, self.is_training: True})
          sample_count += 1
        try:
          all_labels = np.concatenate((all_labels), _labels, axis=0)
          all_predictions = np.concatenate((all_predictions, _predicsions), axis=0)
        except:
          all_labels = _labels
          all_predictions = _predictions
        if sample_count % 10 == 0:                  
          top_one = np.mean(np.argmax(all_labels, 1)==np.argmax(all_predictions, 1))
          all_labels = 0
          all_predictions = 0
          logger.info("Training step: {:05d}, Top one accuracy: {:0.5f}, loss: {:0.5f}.".format(sample_count, top_one, _loss))
          self.training_time = time.time()-start_training
          elapsed_time = time.time()-self.start
          logger.info("Training time: {:.1f}, test time: {:.1f}, elapsed_time: {:.1f}, sum: {:.1f}.".format(self.training_time, self.test_time, elapsed_time, self.training_time+elapsed_time+self.test_time))
          if self.training_time+elapsed_time+self.test_time+15 > self.time_budget:
            self.done_training = True
            break
          start_training = time.time()
        else:
          a = 2

      except tf.errors.OutOfRangeError:
        print("tf out of range Error")
        break
      self.seen_sample += self.batch_size      
    logger.info("Number of training examples: {}".format(sample_count))

  def test(self, dataset, remaining_time_budget=None):
    start = time.time()
    if self.test_iterator_handle is None:
    
      # init dataset:  
      if self.metadata.get_tensor_shape()[0] != -1:
        self.batch_size = 4*128
        TfResize = lambda x: tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(x, 0), self.input_shape), 0)
        def mapFunction(x, y):
          return tf.map_fn(TfResize, x), y
        dataset = dataset.map(mapFunction)
      else: 
        def mapFunction(x, y):
          return tf.map_fn(self.preprocess_tensor_4d, tf.expand_dims(x, 0)), y
        dataset = dataset.map(mapFunction)     
      
      dataset = dataset.batch(int(self.batch_size)).prefetch(1)
      #dataset = dataset.repeat(0)
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
        print("sample_count "+ str(sample_count))
      except tf.errors.OutOfRangeError:
        break
    logger.info("Number of test examples: {}".format(sample_count))
    #import ipdb;ipdb.set_trace()
    predictions = all_predictions[1:, :]
    self.test_time = time.time()-start
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
