from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from glob import glob
from PIL import Image

from keras import optimizers
import pandas as pd

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs_Train(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
#    n = min([len(digit_indices[d]) for d in range(1,4)]) - 1
    n=50
#    n=50
 
    for d in range(2):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1,2)

            dn = (d + inc) % 2
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_pairs_Test(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
#    n = min([len(digit_indices[d]) for d in range(1,4)]) - 1
  
    n=22
#    n=30
    for d in range(2):
        
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1,2)
            dn = (d + inc) % 2
#            if(dn==0):
#                dn=dn+1
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
    

def create_pairs_Test1(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
#    n = min([len(digit_indices[d]) for d in range(1,4)]) - 1
  
    n=22
#    n=30
    for d in range(1):
        d=d+1
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            labels += [1]
    return np.array(pairs), np.array(labels)
    



def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
#    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.25))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.25))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dense(128, activation='relu'))
#    seq.add(Dropout(0.25))
    
    seq.add(Dense(128, activation='relu'))
#    seq.add(Dropout(0.1))
    
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    
    return labels[predictions.ravel() < 0.5].mean()


def create_dataset(directory,fname=""):
    dataset=[]
    for file_count,file_name in enumerate(sorted(glob(directory),key=len)):
#        image=Image.open(file_name)
        img=Image.open(file_name).convert('LA') #to gray scale
        pixels=[f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    if len(fname)>0:
        df=pd.read_csv(fname)
        return np.array(dataset),np.array(df["Class"])
    else:
        return np.array(dataset)

train_x,train_y=create_dataset("D:/Papers/Code/Train/*.jpg","D:/Papers/Code/train.csv")

test_x,test_y=create_dataset("D:/Papers/Code/Test/*.jpg","D:/Papers/Code/test.csv")




# the data, shuffled and split between train and test sets

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train)=(train_x,train_y);
(x_test,y_test)=(test_x,test_y);

#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)

#x_train = x_train.reshape(x_train.shape[0],6040)
#
#x_test = x_test.reshape(x_test.shape[0],6040)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


input_dim = 10000

nepochs = 5

b_size=1024

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(2)]
tr_pairs, tr_y = create_pairs_Train(x_train, digit_indices)

#print(tr_pairs.shape)

digit_indices = [np.where(y_test == i)[0] for i in range(2)]
#te_pairs, te_y = create_pairs_Test(x_test, digit_indices)
te_pairs, te_y = create_pairs_Test1(x_test, digit_indices)



# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)


# train
#rms = RMSprop()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)

opt=sgd
model.compile(loss=contrastive_loss, optimizer=opt)


#print(tr_pairs[:, 0])

#model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#          batch_size=b_size,
#          nb_epoch=nepochs,
#          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=b_size,
          nb_epoch=nepochs,
          validation_data=([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y))

# compute final accuracy on training and test sets

pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)

pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])

te_acc = compute_accuracy(pred, te_y)


print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


