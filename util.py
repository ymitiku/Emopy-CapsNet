import tensorflow as tf
from keras.layers import Layer, Reshape,Dense
from keras.layers import Conv2D,Input
# import tensorflow as tf
import keras
import numpy as np
import os
from keras import backend as K
from keras.models import Model
from sklearn.utils import shuffle
import cv2
import dlib
from keras.preprocessing.image import ImageDataGenerator
EMOTIONS = {
    0 : 'anger', 
    1 : 'disgust', 
    2 : 'fear', 
    3 : 'happy', 
    4 : 'sad', 
    5 : 'surprise', 
    6 : 'neutral'
}
# IMG_SIZE  = (28,28)
class Length(Layer):

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
class CapsLayer(Layer):
    def __init__(self, num_output = 32,
        batch_size=32,length_dim = 8,num_caps = None,layer_type="pcap",num_rout_iter=3,**kwargs):
        """
        :param num_caps: number capsules in this layer
        :param length_dim: dimension of capsules output length
        :param layer_type: type of layer either primary capsule layer(pcap) or capsule layer(cap).
        """
        super(CapsLayer, self).__init__(**kwargs)
        self.num_output = num_output
        self.length_dim = length_dim
        self.layer_type = layer_type
        self.num_rout_iter = num_rout_iter
        self.num_caps = num_caps
        
    def call(self,input,kernel_size=[9,9],strides = 2,padding="valid"):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        if (self.layer_type == "pcap"):
            capsules = []
            for i in range(self.num_output):
                caps_i = Conv2D(self.length_dim,kernel_size=self.kernel_size,strides=self.strides,
                    activation="relu",padding=self.padding,name="conv_"+str(i))(input)
                caps_i_shape = caps_i.shape.as_list()
                caps_i = K.reshape(caps_i,(-1,caps_i_shape[1] * caps_i_shape[2],self.length_dim))
                capsules.append(caps_i)
            capsules_shape = capsules[0].shape.as_list()
            print capsules_shape,"primary caps"
            self.num_caps = capsules_shape[1] * self.num_output
            capsules = keras.layers.concatenate(capsules, axis=1)
            return capsules
            
                
        elif (self.layer_type == "cap"):
            # input.shape (-1,cpa)
            self.net_input = input
            caps = self.routing(self.net_input)
            return caps
        else:
            raise Exception("Not implmented for "+str(self.layer_type))
    def routing(self,input):
        
        # input shape None,num_caps,input_length_dim
        input = K.expand_dims(input,axis=2)
        input = K.expand_dims(input,axis=3)
        # None,input_num_caps,1,1,input_length_dim
        input = K.tile(input,[1,1,self.num_caps,1,1])
        # input shape (?, input_num_caps,self.num_caps,1,input_length_dim)
        # weight shape (32, 32, 6, 6,10,8,16)\
        print input.shape
        input_shape = input.shape.as_list()
        weight_shape = [input_shape[1],self.num_caps,input_shape[4],self.length_dim]
        self.W = self.add_weight(shape=weight_shape,
                                 initializer='glorot_uniform',
                                 name='W')
        self.b_IJ = self.add_weight(shape=[1,input_shape[1],self.num_caps,1,1],
                                 initializer="zeros",
                                 name='bias',
                                 trainable=False
                                 )

        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=input,
                             initializer=K.zeros([input_shape[1], self.num_caps, 1, self.length_dim]))

        print "uhat ", inputs_hat.shape
        # print "uhat", u_hat.shape
        for iter in range(self.num_rout_iter):
            # b_IJ shape b_J (1, 1152, 10, 1, 1)
            c_IJ = tf.nn.softmax(self.b_IJ, dim=2)
            s_J = K.sum(c_IJ * inputs_hat, 1, keepdims=True)
            v_J = self.squash(s_J)
            if iter!=self.num_rout_iter-1:
                self.b_IJ += K.sum(inputs_hat * v_J, -1, keepdims=True)

        v_J = K.reshape(v_J, [-1, self.num_caps, self.length_dim])
        return v_J
    def squash(self,vector):
        vec_squared_norm = K.sum(K.square(vector),axis = -1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / K.sqrt(vec_squared_norm)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)
    def compute_output_shape(self,input_shape):
        return tuple([None, self.num_caps, self.length_dim])

class CapsNet(object):

    def __init__(self,input_shape,lmd = 0.5,learing_rate = 1e-4):
        
        self.input = Input(shape=input_shape)
        conv1 = Conv2D(32,activation="relu",kernel_size=[9,9],strides=1,padding="valid",name="conv1")(self.input)
        primaryCaps = CapsLayer(length_dim=8)(conv1,padding="valid")
        secondCaps = CapsLayer(num_caps = len(EMOTIONS),length_dim = 16,layer_type="cap")(primaryCaps)
        length = Length(name="pred")(secondCaps)
        self.model = Model(inputs=self.input,outputs=length)
        self.learing_rate = learing_rate
        self.lmd = lmd
        self.input_shape = input_shape
    def train(self):
        
        self.x_train, self.y_train = self.load_dataset("/home/mtk/iCog/projects/emopy/dataset/all/train",True)
        self.x_test, self.y_test = self.load_dataset("/home/mtk/iCog/projects/emopy/dataset/all/test",True)

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = shuffle(self.x_test,self.y_test)
        
        x_train = self.x_train.reshape((-1,self.input_shape[0],self.input_shape[1],1))
        x_test = self.x_test.reshape((-1,self.input_shape[0],self.input_shape[1],1))
        y_train = np.eye(len(EMOTIONS))[self.y_train]
        y_test = np.eye(len(EMOTIONS))[self.y_test]
     
        self.model.compile(optimizer=keras.optimizers.Adam(self.learing_rate),
                  loss=[self.margin_loss],
                  metrics=['accuracy'])
        self.model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=32)

        # datagen = ImageDataGenerator(
        #     rotation_range=30,
        #     width_shift_range = 0.2,
        #     height_shift_range = 0.2,
        #     shear_range = 0.2,
        #     zoom_range = 0.2,
        #     horizontal_flip = True,
        # )
        # self.model.fit_generator( datagen.flow(x_train,y_train, batch_size=32), 
        #                     steps_per_epoch = 1000,
        #                     validation_data =(x_test,y_test),
        #                     verbose = 1,
        #                     epochs = 100
        # )
        self.model.save_weights("models/all-model.h5")
    def string_to_emotion(self,string):
        for emotion in EMOTIONS:
            if EMOTIONS[emotion] == string:
                return emotion
        raise Exception("value "+string," does not exist")
    def sanitize(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (self.input_shape[0],self.input_shape[1]))  # Resize
        return img

    def load_dataset(self,directory,verbose=True):
        x, y = [], []

        # Read images from the directory
        for emotion_dir in os.listdir(directory):
            if verbose:
                print "loading",emotion_dir,"dataset"
            for filename in os.listdir(os.path.join(directory, emotion_dir)):
                try:
                    x += [self.sanitize(cv2.imread(os.path.join(directory, emotion_dir, filename)))]
                except cv2.error,e:
                    print "Error while reading ", os.path.join(directory, emotion_dir, filename)
                    continue
                y += [self.string_to_emotion(emotion_dir)]
                # y +=[EMOTIONS(emotion_dir)]
                

        # Convert to numpy array
        x = np.array(x, dtype='uint8')
        y = np.array(y)

        return x, y
    def margin_loss(self,y_true, y_pred):
       
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            self.lmd * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

        
