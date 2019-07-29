import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import cv2
# In[]:
classes_List=['A','O']
numOfClasses=len(classes_List)
    
# In[]
def get_data(size, total_sample_size):
    
    image = cv2.imread('Train/'+classes_List[0] +'/'+ str(classes_List[0])+ str(1)+'.jpg')
    
    #get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0
    
    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2]) # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    
    for i in range(numOfClasses):
        for j in range(int(total_sample_size/numOfClasses)):
            ind1 = 0
            ind2 = 0
            
            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            img1 = cv2.imread('Train/'+classes_List[i] +'/'+ str(classes_List[i])+ str(ind1+1)+'.jpg')
            img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            
            img2 = cv2.imread('Train/'+classes_List[i]+'/'+str(classes_List[i]) + str(ind2+1)  + '.jpg')
            img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/10)):
        for j in range(10):
            
            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(numOfClasses)
                ind2 = np.random.randint(numOfClasses)
                if ind1 != ind2:
                    break
             
            img1 = cv2.imread('Train/'+classes_List[ind1]+'/'+classes_List[ind1]+ str(j+1) + '.jpg')
            img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            
            img2 = cv2.imread('Train/'+classes_List[ind2]+'/'+classes_List[ind2]+ str(j+1) + '.jpg')
            img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1
            
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y                          


# In[]:
size=2
total_sample_size=10000

X, Y = get_data(size, total_sample_size)

print(X.shape)
print(Y.shape)

# In[]:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

# In[]:
def build_base_network(input_shape):
    
    seq = Sequential()
    
    nb_filter = [6, 12]
    kernel_size = 3
    
    
    #convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    #convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))

    #flatten 
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq
# In[]:
input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

# In[]:
    
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

# In[]:

epochs = 13
rms = RMSprop()

model = Model(input=[img_a, img_b], output=distance)

# In[]:
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

model.compile(loss=contrastive_loss, optimizer=rms)

# In[]:
img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, nb_epoch=epochs)

# In[]:
pred = model.predict([x_test[:, 0], x_test[:, 1]])

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

# In[]          
compute_accuracy(pred, y_test)
    


