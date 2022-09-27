import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

import numpy as np
import math

l2 = tf.keras.regularizers.l2(4e-5)
initializer = tf.random_normal_initializer(stddev=0.01)


def conv(x, output_c, k=1,s=1, p='same'):
  #if s == 1:
  #  padding = 'same'
  #else:
  #if s==2:
  #  x = layers.ZeroPadding2D(((0, 1), (0, 1)))(x)
  #  padding = 'valid'
        
  x=Conv2D(output_c,k,s,padding=p,kernel_regularizer=l2)(x)
  x=keras.layers.BatchNormalization(epsilon=0.001)(x)
  #x=keras.activations.swish(x)
  x=keras.activations.selu(x)

  return x


def bottleneck(x,input_c,output_c):
  c=input_c//2

  return x+conv(conv(x,c),output_c,k=3)


def cspBottleneck(x,input_c,output_c,n):
  c=input_c//2
  y1=conv(x,c)
  y2=conv(x,c)
  
  for i in range(n):
    y2=bottleneck(y2,c,c)

  return conv(tf.concat([y1,y2],axis=-1),output_c)


def spp(x,input_c,output_c,k=(5,9,13)):
  c=input_c//2
  y1=conv(x,c,1)
  y2=[tf.nn.max_pool(y1,ksize=i,strides=1,padding='SAME') for i in k]

  y3=tf.concat(([y1]+y2),axis=-1)
  y4=conv(y3,output_c)

  return y4


def build_model(input_size,width,depth,nc):
  #input=keras.Input(shape=(416,416,3))
  input=keras.Input(shape=(input_size,input_size,3))
  x=input

  x=conv(x,int(round(64*width)),k=6,s=2)
  x=conv(x,int(round(128*width)),3,2)
  x=cspBottleneck(x,int(round(128*width)),int(round(128*width)),int(round(3*depth)))

  x=conv(x,int(round(256*width)),k=3,s=2)
  x=cspBottleneck(x,int(round(256*width)),int(round(256*width)),int(round(6*depth)))
  y1=x

  x=conv(x,int(round(512*width)),3,2)
  x=cspBottleneck(x,int(round(512*width)),int(round(512*width)),int(round(9*depth)))
  y2=x

  x=conv(x,int(round(1024*width)),3,2)
  x=cspBottleneck(x,int(round(1024*width)),int(round(1024*width)),int(round(3*depth)))

  x=spp(x,int(round(1024*width)),int(round(1024*width)))

  x=conv(x,int(round(512*width)))
  y3=x

  x=layers.UpSampling2D()(x)
  x=cspBottleneck(tf.concat([x,y2],axis=-1),int(round(1024*width)),int(round(512*width)),int(round(3*depth)))
  y4=x

  x=layers.UpSampling2D()(x)
  x=cspBottleneck(tf.concat([x,y1],axis=-1),int(round(512*width)),int(round(256*width)),int(round(3*depth)))
  y5=x

  r1=Conv2D(3*(nc+5),kernel_size=1,kernel_initializer=initializer, kernel_regularizer=l2)(y5)

  x=conv(x,int(round(256*width)),k=3,s=2)
  x=cspBottleneck(tf.concat([x,y4],axis=-1),int(round(512*width)),int(round(512*width)),int(round(3*depth)))
  y6=x

  r2=Conv2D(3*(nc+5),kernel_size=1,kernel_initializer=initializer, kernel_regularizer=l2)(y6)

  x=conv(x,int(round(512*width)),k=3,s=2)
  x=cspBottleneck(tf.concat([x,y3],axis=-1),int(round(512*width)),int(round(512*width)),int(round(9*depth)))

  r3=Conv2D(3*(nc+5),kernel_size=1,kernel_initializer=initializer, kernel_regularizer=l2)(x)

  output=[tf.reshape(r1,[-1,3,r1.shape[1],r1.shape[2],5+nc]),tf.reshape(r2,[-1,3,r2.shape[1],r2.shape[2],5+nc]),tf.reshape(r3,[-1,3,r3.shape[1],r3.shape[2],5+nc])]

  return Model(input,output)


def get_model(img_size,nc, width=1, depth=1, lr=0.001):
  model=build_model(img_size,width,depth,nc)
  grids=np.array([model.output_shape[0][2],model.output_shape[1][2],model.output_shape[2][2]])
  opt=keras.optimizers.Adam(learning_rate=lr)
  model.compile(optimizer=opt)

  return model, grids
