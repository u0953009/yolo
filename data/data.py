import os
import cv2
import numpy as np
import json

def build_coco(img_path,label_path):
  filename_id={}  # {{'filename' : id},...}
  id_boxes={}     # {{id : [[x,y,w,h,cls],...]},...}
  image_wh={}     # {{id : [w,h,w,h]}...}   
  labels={}       # {{'filename' : []},...}  this dictionary just to check if the corresponding image to the current box is not missing

  la=json.load(open(label_path))
  filenames=os.listdir(img_path)
  for fn in filenames:
    labels[fn]=[]
  for img in la['images']:
    if img['file_name'] in labels:
      filename_id[img['file_name']]=img['id']
      id_boxes[img['id']]=[]
      image_wh[img['id']]=[img['width'],img['height'],img['width'],img['height']]
      
      
  for annotation in la['annotations']:
    if annotation['image_id'] in id_boxes:
      id_boxes[annotation['image_id']].append( annotation['bbox']+[annotation['category_id']])

  return filenames,filename_id,id_boxes,image_wh

def build_data_coco(filenames, filename_id,id_boxes,image_wh,idx,batch_size,file_path,input_size, normalize=True):
  batch_=filenames[batch_size*idx:np.min([batch_size*idx+batch_size, len(filenames)])]
  batch=[]
  target=[]
  for i in range(len(batch_)):
    img_id=filename_id[batch_[i]]
    if img_id in id_boxes:
      image_path=file_path+batch_[i]
      
      batch.append(cv2.resize(cv2.imread(image_path),(input_size,input_size)))
      
      #img_size=np.concatenate((np.array(image_wh[filename_id[batch_[i]]]),np.array(image_wh[filename_id[batch_[i]]]))  )
      img_size=np.array(image_wh[img_id])

      box=np.array(id_boxes[img_id])
      #print(box)
      if len(box)>0: 
        class_id=box[...,4]-1
        box_coord=box[...,:4]
        #print(i)
        #print(box[...,:4])
        #print(img_size)
        #print(box[...,:4]/img_size)
        bboxes=np.concatenate( (  np.ones(len(box))[...,None]*i,class_id[...,None],box[...,:4]/img_size) ,-1) #number in batch, class, x,y,w,h
        target.append(bboxes)
  
  if normalize:
    if len(target)==0:
      return (np.array(batch, dtype=np.float32))/255, np.array([])    
    return (np.array(batch, dtype=np.float32))/255, np.concatenate(target,0)  

  if len(target)==0:
      return np.array(batch), np.array([])  
  return np.array(batch), np.concatenate(target,0)
  
  


# return : a batch of normalized input, a batch of (class, x1,y1,w,h) in ratio
# input label format : class, Xcenter, Ycenter, W, H
def build_data(filenames,img_path,label_path,start_idx,batch_size,input_size,normalize=True):
  nn=filenames[start_idx: np.minimum(start_idx+batch_size,len(filenames))]  
  batch=[]
  target=[]
  for i in range(len(nn)):
    batch.append(cv2.resize(cv2.imread(img_path+nn[i]),(input_size,input_size)))
    with open(label_path+nn[i].split('.')[0]+'.txt') as label:
      l=label.readline()
      while l:
        tmp=l.split(' ')
        t=[i,tmp[0],np.maximum(float(tmp[1])-float(tmp[3])/2,0),np.maximum(float(tmp[2])-float(tmp[4])/2,0),float(tmp[3]),float(tmp[4])]
        target.append(t)
        l=label.readline()
  if normalize:
    return np.array(batch)/255, np.array(target).astype(float)
  return np.array(batch), np.array(target).astype(float)


def build_data_predict(filenames,img_path,start_idx,batch_size,input_size,normalize=True):
  nn=filenames[start_idx: np.minimum(start_idx+batch_size,len(filenames))]  
  batch=[]
  img_size=[]
  
  for i in range(len(nn)):
    img=cv2.imread(img_path+nn[i])
    img_size.append((img.shape[1],img.shape[0]))    
    batch.append(cv2.resize(img,(input_size,input_size)))

  if normalize:
    return np.array(batch)/255, img_size
  return np.array(batch), img_size
