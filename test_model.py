import os,sys
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from matplotlib import pyplot as plt
from skimage import io,data,transform
from PIL import Image,ImageDraw,ImageFont
import numpy as np
#np.random.seed(1)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf
#tf.set_random_seed(2)
#tf.random.set_seed(2)
#tf.random.set_seed(args.seed)
from keras.backend import conv1d
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
# from keras.layers.convolutional import Conv1D, MaxPooling1D,AveragePooling1D
from keras.layers import Dense, LSTM,GRU, MaxPooling1D,Flatten,Conv1D

from keras.layers import Embedding, SimpleRNN

from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from keras.optimizers import SGD
from keras.optimizers import Adam
import os
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from matplotlib.font_manager import FontProperties
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import os,sys
os.getcwd()
os.chdir("./GRU")
print(os.getcwd())
print (sys.version)

def load_data_test_1():
    maxlen=60
    x_test = np.loadtxt('./data/test1.csv', delimiter=',')[:, 1:]
    y_test = np.loadtxt('./data/test1.csv', delimiter=',')[:, 0]

    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / \
    (y_test.max()-y_test.min())*(nb_classes-1)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    x_test = sequence.pad_sequences(x_test,maxlen=maxlen)

    return x_test, Y_test


x_test,Y_test=load_data_test_1()
print (x_test.shape,'----x_test.shape')
print (Y_test.shape,'----Y_test.shape')
model = keras.Sequential()
MODEL_PATH = './GRU_model'
model = load_model(MODEL_PATH)
loss_and_acc = model.evaluate(x_test, Y_test)
print('loss= '+str(loss_and_acc[0]))
print('acc= '+str(loss_and_acc[1]))
test_pred=model.predict(x_test)
confm = metrics.confusion_matrix(np.argmax(test_pred,axis=1),np.argmax(Y_test,axis=1))


   
## 混淆矩阵可视化
Labname = ["strong","middle","weak","no"]
plt.figure(figsize=(8,8))
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False,linewidths=.8,
            cmap="YlGnBu")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('True label',size = 14)
plt.ylabel('Predicted label',size = 14)
plt.xticks(np.arange(10)+0.5,Labname,fontproperties = fonts,size = 12)
plt.yticks(np.arange(10)+0.3,Labname,fontproperties = fonts,size = 12)
plt.show()


print(metrics.classification_report(np.argmax(test_pred,axis=1),np.argmax(Y_test,axis=1)))