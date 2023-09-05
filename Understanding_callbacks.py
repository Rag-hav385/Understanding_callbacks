#!/usr/bin/env python
# coding: utf-8

# <pre>
# 1. Download the data from <a href='https://drive.google.com/file/d/15dCNcmKskcFVjs7R0ElQkR61Ex53uJpM/view?usp=sharing'>here</a>
# 
# 2. Code the model to classify data like below image
# 
# <img src='https://i.imgur.com/33ptOFy.png'>

# <pre>
# <b>Model-1</b>
# <pre>
# 1. Use tanh as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use RandomUniform(0,1) as initilizer.
# 3. Analyze your output and training process. 
# </pre>
# </pre>
# <pre>
# <b>Model-2</b>
# <pre>
# 1. Use relu as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use RandomUniform(0,1) as initilizer.
# 3. Analyze your output and training process. 
# </pre>
# </pre>
# <pre>
# <b>Model-3</b>
# <pre>
# 1. Use relu as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use he_uniform() as initilizer.
# 3. Analyze your output and training process. 
# </pre>
# </pre>
# <pre>
# <b>Model-4</b>
# <pre>
# 1. Try with any values to get better accuracy/f1 score.  
# </pre>
# </pre>

# # libraries

# In[1]:


import pandas as pd
import numpy as np
import datetime

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model
import random as rn

from sklearn.metrics import f1_score , auc ,roc_auc_score


# # Getting data and loading the dataset

# In[2]:


data = pd.read_csv("data.csv")
X = data.drop("label" , axis = 1)
Y = data["label"]
print("="*25)
print(X.shape)
print(Y.shape)


# In[3]:


X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.33 , random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[4]:


data["label"].value_counts()


# # Simple Model for practice

# In[20]:


#Input layer
input_layer = Input(shape=(2,))
#Dense hidden layer
layer1 = Dense(50,activation='sigmoid',kernel_initializer=tf.keras.initializers.glorot_normal(seed=30))(input_layer)
#output layer
output = Dense(2,activation='softmax',kernel_initializer=tf.keras.initializers.glorot_normal(seed=0))(layer1)
#Creating a model
model = Model(inputs=input_layer,outputs=output)





optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=3, validation_data=(X_test,Y_test), batch_size=16)


# # Callbacks

# In[18]:


from tensorflow.keras.callbacks import ModelCheckpoint
import keras
class TerminateNaN(tf.keras.callbacks.Callback):
        
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print("Invalid loss and terminated at epoch {}".format(epoch))
                self.model.stop_training = True

Terminate_NaN = TerminateNaN()

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

def changeLearningRate(epoch):
    initial_learningrate=0.01
    changed = initial_learningrate
    if (epoch-1)%3 == 0:
        changed = initial_learningrate*(1-0.005)**epoch
    else:
        changed = initial_learningrate*(1-0.001)**epoch
    return changed


# In[19]:


changed_lr = []
for i in range(1,50):
    changed_lr.append(changeLearningRate(i))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(changed_lr)
plt.ylabel('learning_rate')
plt.xlabel('epoch number')


# In[11]:


from google.colab import drive
drive.mount('/content/drive')


# In[10]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


rm -rf . 


# <b>F1 Score and AUC score</b> 

# In[24]:


#Reference:https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2

class return_F1_and_AUC_score(tf.keras.callbacks.Callback):

    def  __init__(self , validation_x , validation_y):
        self.validation_x = validation_x
        self.validation_y = validation_y
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_aucs = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        val_predict_round = (np.asarray(self.model.predict(self.validation_x))).round()
        val_predict = np.asarray(self.model.predict(self.validation_x))
        val_targ = list(self.validation_y)
        
        #print(val_predict)
        f1_sc = f1_score(val_predict_round , val_targ)
        auc_sc = 0
        try:
            auc_sc = roc_auc_score(val_predict_round , val_targ)
        except ValueError:
            auc_sc = 0
        
        self.val_f1s.append(f1_sc)
        self.val_aucs.append(auc_sc)
        print(" F1-Score:{} , AUC-score:{}".format(f1_sc , auc_sc))
    
    
F1_and_AUC = return_F1_and_AUC_score(X_test, Y_test)


# # Model-1

# <b>Model-1</b>
# <pre>
# 1. Use tanh as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use RandomUniform(0,1) as initilizer.
# 3. Analyze your output and training process. 
# </pre>

# In[15]:


#Input layer
input_layer = Input(shape=(2,))
#Dense hidden layer1
layer1 = Dense(100,activation='tanh',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(input_layer)

#Dense hidden layer2
layer2 = Dense(80,activation='tanh',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer1)

#Dense hidden layer3
layer3 = Dense(50,activation='tanh',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer2)

#Dense hidden layer4
layer4 = Dense(30,activation='tanh',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer3)

#Dense hidden layer5
layer5 = Dense(20,activation='tanh',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer4)

#output layer
output = Dense(1,activation="sigmoid",kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer5)
#Creating a model
model = Model(inputs=input_layer,outputs=output)

#Callbacks
F1_and_AUC = return_F1_and_AUC_score(X_test, Y_test)
filepath="model_save/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9,patience=0, min_lr=0.001)
lrschedule = LearningRateScheduler(changeLearningRate, verbose=0.1)
Terminate_NaN = TerminateNaN()
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.20, patience=2, verbose=1)

log_dir="logs1\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

#callbacksList
callback_list = [F1_and_AUC , checkpoint , reduce_lr , lrschedule, Terminate_NaN , earlystop,tensorboard_callback]

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,nesterov=True)

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])

model.fit(X_train,Y_train,epochs=6,validation_split=0.1, validation_data = (X_test,Y_test), batch_size=16 , callbacks=callback_list)


# In[ ]:


get_ipython().getoutput('kill 2159')


# In[16]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# #model-2

# <b>Model-2</b>
# <pre>
# 1. Use relu as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use RandomUniform(0,1) as initilizer.
# 3. Analyze your output and training process. 
# </pre>

# In[25]:


input_layer = Input(shape=(2,))
#Dense hidden layer1
layer1 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(input_layer)

#Dense hidden layer2
layer2 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer1)

#Dense hidden layer3
layer3 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer2)

#Dense hidden layer4
layer4 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer3)

#Dense hidden layer5
layer5 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer4)

#output layer
output = Dense(1,activation="sigmoid",kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.))(layer5)
#Creating a model
model = Model(inputs=input_layer,outputs=output)

#Callbacks
F1_and_AUC = return_F1_and_AUC_score(X_test, Y_test)
filepath="model_save/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9,patience=0, min_lr=0.001)
lrschedule = LearningRateScheduler(changeLearningRate, verbose=0.1)
Terminate_NaN = TerminateNaN()
earlystop = EarlyStopping(monitor='val_auc', min_delta=0.20, patience=2, verbose=1)

log_dir="logs2\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

#callbacksList
callback_list = [F1_and_AUC , checkpoint , reduce_lr , lrschedule, Terminate_NaN , earlystop,tensorboard_callback]

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,nesterov=True)

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])

model.fit(X_train,Y_train,epochs=6, validation_data = (X_test,Y_test), batch_size=16 , callbacks=callback_list)


# In[27]:


get_ipython().system('kill 506')


# In[28]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# #Module3

# <b>Model-3</b>
# <pre>
# 1. Use relu as an activation for every layer except output layer.
# 2. use SGD with momentum as optimizer.
# 3. use he_uniform() as initilizer.
# 3. Analyze your output and training process. 
# </pre>

# In[29]:


input_layer = Input(shape=(2,))
#Dense hidden layer1
layer1 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(input_layer)

#Dense hidden layer2
layer2 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer1)

#Dense hidden layer3
layer3 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer2)

#Dense hidden layer4
layer4 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer3)

#Dense hidden layer5
layer5 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer4)

#output layer
output = Dense(1,activation="sigmoid",kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer5)
#Creating a model
model = Model(inputs=input_layer,outputs=output)

#Callbacks
F1_and_AUC = return_F1_and_AUC_score(X_test, Y_test)
filepath="model_save2/weights-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,patience=0, min_lr=0.001)
lrschedule = LearningRateScheduler(changeLearningRate, verbose=0.1)
Terminate_NaN = TerminateNaN()
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.20, patience=2, verbose=1)

log_dir="logs3\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

#callbacksList
callback_list = [F1_and_AUC , checkpoint , reduce_lr , lrschedule, Terminate_NaN , earlystop,tensorboard_callback]

optimizer = tf.keras.optimizers.SGD()

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=6,  validation_split=0.1 ,validation_data = (X_test,Y_test), batch_size=16 , callbacks=callback_list)


# In[31]:


get_ipython().system('kill 1224')


# In[32]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# In[ ]:


pip install -U tensorboard-plugin-profile


# #Model-4

# In[33]:


input_layer = Input(shape=(2,))
#Dense hidden layer1
layer1 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(input_layer)

#Dense hidden layer2
layer2 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer1)

#Dense hidden layer3
layer3 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer2)

#Dense hidden layer4
layer4 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer3)

#Dense hidden layer5
layer5 = Dense(50,activation='relu',kernel_initializer=tf.keras.initializers.he_uniform(seed = 30))(layer4)

#output layer
output = Dense(1,activation="sigmoid",kernel_initializer=tf.keras.initializers.glorot_normal(seed=30))(layer5)
#Creating a model
model = Model(inputs=input_layer,outputs=output)

#Callbacks
F1_and_AUC = return_F1_and_AUC_score(X_test, Y_test)
filepath="model_save2/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.01,patience=0, min_lr=0.001)
lrschedule = LearningRateScheduler(changeLearningRate, verbose=0.1)
Terminate_NaN = TerminateNaN()
earlystop = EarlyStopping(monitor='loss', min_delta=0.20, patience=2, verbose=1)

log_dir="logs4\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

#callbacksList
callback_list = [F1_and_AUC , checkpoint , reduce_lr , lrschedule, Terminate_NaN , earlystop,tensorboard_callback]

optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])

model.fit(X_train,Y_train,epochs=6,  validation_split=0.1 ,validation_data = (X_test,Y_test), batch_size=16 , callbacks=callback_list)


# In[35]:


get_ipython().system('kill 1827')


# In[36]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# In[ ]:


get_ipython().system('pwd')

