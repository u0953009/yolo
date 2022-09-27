import argparse
import tensorflow as tf
from data import models, util, data
import numpy as np
import os
import cv2
import math






def draw_batch(xyxy,fn,path,result_path):
  
  for i in range(len(xyxy)):
    img=cv2.imread(path+fn[i])
    
    if len(xyxy[i])>0:
      boxes=np.array(xyxy[i])[...,:4][0]
      
      h,w=img.shape[0],img.shape[1]
      
      x1,y1,x2,y2=boxes[...,0]*w, boxes[...,1]*h,boxes[...,2]*w, boxes[...,3]*h
      
      for j in range(len(xyxy[i])):
        cv2.rectangle(img,(int(x1[j]),int(y1[j])),(int(x2[j]),int(y2[j])),(255,255,255),2)
    cv2.imwrite(result_path+fn[i],img)
  
  
  
 

def predict(model,img_path,result_path,input_size,grids,anchors,nl, conf, iou):
    filenames=os.listdir(img_path)
    batch_size=96

    step=math.ceil(len(filenames)/96)
    for i in range(step):
        batch,size=data.build_data_predict(filenames,img_path,i*batch_size,batch_size,input_size,normalize=True)

        #batch2,size2=data.build_data_predict(filenames,img_path,i*batch_size,batch_size,input_size,normalize=False)
        y_pred=model(batch)
        
        boxes=util.get_boxes(y_pred,input_size, grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=iou)
        draw_batch(boxes,filenames[i*batch_size:i*batch_size+len(batch)],img_path,result_path)
        
        

def args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--depth', type=float, default=1.0)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--nc',type=int, default=90)
    parser.add_argument('--lr',type=float, default=0.001)
    args=parser.parse_args()

    return args
  
def main():
    arg=args()
    img_path=arg.img_path
    result_path=arg.result_path
    width=arg.width
    depth=arg.depth
    input_size=arg.input_size
    conf=arg.conf
    iou=arg.iou
    nc=arg.nc
    lr=arg.lr
    na=3
    nl=3

    anchors=np.array([[[10.0,13.0], [16.0,30.0], [33.0,23.0]]  ,# P3/8
                        [[30.0,61.0], [62.0,45.0], [59.0,119.0]]  ,# P4/16
                        [[116.0,90.0], [156.0,198.0], [373.0,326.0]] ],dtype='float32') # P5/32
    anchors*=input_size/640
    
    model,grids=models.get_model(input_size,nc,width,depth,lr)
    model.load_weights(arg.weights)

    predict(model,img_path,result_path, input_size,grids,anchors,nl, conf,iou)

    

    

if __name__=="__main__":
    
    main()

