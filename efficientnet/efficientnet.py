
import tensorflow as tf
import efficientnet_builder as efficientnet_builder

class EfficientNet(object):

    def __init__(self,model_name='efficientnet-b0',num_classes=1000,skip_load=[]):

        self.scores=None
        self.model_name = model_name
        self.skip_load_=skip_load
        self.params={"num_classes":num_classes}

        _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
        self.scale_size_=(image_size,image_size)

        print("EfficientNet model %s size: %s"%(model_name,str(self.scale_size_)))

    def model(self,x,is_training):

        #is_train = (is_training == tf.estimator.ModeKeys.TRAIN)
        is_train=True
        logits, endpoints = efficientnet_builder.build_model(x, self.model_name, is_train,override_params=self.params)
        self.scores= logits
        return logits, endpoints

    def score(self):
        return self.scores

    def scale_size(self):
        return self.scale_size_

    def init(self, sess):
        return

    def skip_load(self):
        return self.skip_load_

