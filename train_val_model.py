import numpy as np
np.random.seed(1)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf
from keras.backend import conv1d
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, GRU, MaxPooling1D,Flatten,Conv1D

from keras.layers import Embedding, SimpleRNN

from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
import pickle

from keras.optimizers import SGD
from keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.initializers import RandomNormal

#数据导入
maxlen=60
x_train = np.loadtxt('./data/data.csv',dtype=float,delimiter=',')[:,1:]
y_train = np.loadtxt('./data/data.csv',dtype=float,delimiter=',')[:,0]
x_test = np.loadtxt('./data/valdata.csv',dtype=float,delimiter=',')[:,1:]
y_test = np.loadtxt('./data/valdata.csv',dtype=float,delimiter=',')[:,0]
nb_classes = len(np.unique(y_train))
y_train = (y_train - y_train.min()) / \
(y_train.max()-y_train.min())*(nb_classes-1)
Y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = (y_test - y_test.min()) / \
(y_test.max()-y_test.min())*(nb_classes-1)
Y_test = np_utils.to_categorical(y_test, nb_classes)
x_train = sequence.pad_sequences(x_train,maxlen=60)
x_test = sequence.pad_sequences(x_test,maxlen=60) 
#GRU模型
epochs=50
model = Sequential()
model.add(Embedding(len(np.unique(x_train))+1, 32, input_length=60))
model.add(Dropout(0.4))
model.add(GRU(16,return_sequences=True))
    # model.add(Dropout(0.2))
model.add(GRU(32))
model.add(Dropout(0.4))
model.add(Dense(5, activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])
    
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.000001)
model.summary()
    # history = model.fit(x_train,Y_train,epochs = epochs,batch_size=64,validation_split=0.2, callbacks=[reduce_lr])
history =model.fit(x_train,Y_train,epochs = epochs,batch_size=64,validation_data=(x_test, Y_test), callbacks=[reduce_lr])
    # history = model.fit(x_train,Y_train,epochs = epochs,batch_size=64,validation_data=(x_test, Y_test), callbacks=[early_stopping])
 
model.save('./data/GRU_model')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) +1)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend()
plt.figure()
plt.plot(epochs,acc,'b',label='训练准确率 ')
plt.plot(epochs,val_acc,'r',label='验证准确率 ')
#plt.title('train and validation acc')
plt.ylabel('训练、验证准确率 Accuracy%')
plt.xlabel('迭代次数')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'b',label='训练损失')
plt.plot(epochs,val_loss,'r',label='验证损失')
    #plt.title('train and validation loss')
plt.ylabel('训练、验证准损失函数')
plt.xlabel('迭代次数')
plt.legend()
plt.show()
history_dict3=history.history

#画出模型对比
import matplotlib.pyplot as plt
acc= history_dict1['acc']

val_acc = history_dict1['val_acc']

loss1 = history_dict1['loss']

val_loss =history_dict1['val_loss']

epochs = range(1,len(acc) +1)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend()
plt.plot(epochs,acc,'y',label='GRU')

plt.plot(epochs,val_loss1,'b' ,linestyle='-.',label='GRU')

plt.legend()
plt.ylabel('训练和验证准确率 Accuracy%')
plt.xlabel('迭代次数')