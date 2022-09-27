
import tensorflow as tf
import numpy as np
from data import data, util


global nc
global grids
global anchors
nl=3


def set_args(n,g,a):
  global nc
  global grids
  global anchors

  nc=n
  grids=g
  anchors=a


def evaluate_tf(boxes, label,iou_threshold):
  

  def per_img(idx):
    rst=np.zeros((nc,3))
    if len(label[idx])>0 and len(boxes[idx])>0:

      t,p=label[idx], boxes[idx][0].numpy()
      rst_img=np.zeros([nc,3]) # TP, FP, TN

      for i in range(len(t)):
        cur_t=t[i]
        no_match=1 # if this true box doesn't find a predicted box
        best_iou=0
        best_box=-1

        for j in range(len(p)):
          cur_p=p[j]
          if cur_p[5]==cur_t[0]:
            iou=box_iou(cur_t[1:],cur_p[:4])
            #print(i,j, iou)
            if iou>iou_threshold:
              no_match=0
              if iou>best_iou:
                best_iou=iou
                best_box=j

        if no_match:
          rst_img[int(cur_t[0])][2]+=1
        else:
          rst_img[int(cur_t[0])][0]+=1
          p=np.delete(p,best_box,axis=0)
      for i in range(len(p)):
        rst_img[int(p[i][5])][1]+=1

      rst+=rst_img
    elif len(label[idx])>0 and len(boxes[idx])==0:
      
      for i in range(len(label[idx])):
        rst[int(label[idx][i][0])][2]+=1
    elif len(label[idx])==0 and len(boxes[idx])>0:
      
      for i in range(len(boxes[idx][0])):

        rst[int(boxes[idx][0][i][5])][1]+=1
    return rst

  _=tf.map_fn(fn=per_img,elems=tf.cast(tf.range(len(boxes)),dtype=tf.int64))
  #_=tf.map_fn(fn=per_img,elems=np.array([0]))
  return tf.reduce_sum(_,0)

# boxes : batch of images * 1* number of box per image * number of elements per box(x,y,w,h,conf,class)
# label : batch of images * number of box per batch * number of elemeents per box(class,x,y,w,h)
def evaluate_tf2(boxes, label,iou_threshold):
  rst=np.zeros([nc,3])

  for idx in range(len(boxes)):
    if len(label[idx])>0 and len(boxes[idx])>0:

      t,p=label[idx], boxes[idx][0]
      
      rst_img=np.zeros([nc,3]) # TP, FP, TN
      for i in range(len(t)):
        cur_t=t[i]
        no_match=1 # if this true box doesn't find a predicted box
        best_iou=0
        best_box=-1

        for j in range(len(p)):
          cur_p=p[j]
          if cur_p[5]==cur_t[0]:
            iou=box_iou(cur_t[1:],cur_p[:4])
            #print(i,j, iou)
            if iou>iou_threshold:
              no_match=0
              if iou>best_iou:
                best_iou=iou
                best_box=j

        if no_match:
          rst_img[int(cur_t[0])][2]+=1
        else:
          rst_img[int(cur_t[0])][0]+=1
          p=np.delete(p,best_box,axis=0)
      for i in range(len(p)):
        rst_img[int(p[i][5])][1]+=1

      rst+=rst_img
    elif len(label[idx])>0 and len(boxes[idx])==0:
      
      for i in range(len(label[idx])):
        rst[int(label[idx][i][0])][2]+=1
    elif len(label[idx])==0 and len(boxes[idx])>0:
      
      for i in range(len(boxes[idx][0])):
        rst[int(boxes[idx][0][i][5])][1]+=1
    

  return rst

def target_val(target,batch_size):
  rst=[[] for i in range(batch_size)]
  for i in range(len(target)):
    rst[int(target[i][0])].append( (target[i][1:]).astype(np.float32) )
  return rst


# input, d : number of class * [TP,FP,TN]
def compute_pr_per_class(d):
  d=tf.cast(d,dtype=tf.float32)
  p_denom=d[...,0]+d[...,1]
  r_denom=d[...,0]+d[...,2]

  # nan to 1
  #p_nan=tf.cast(tf.math.equal(p_denom,0),dtype=tf.float32)
  #r_nan=tf.cast(tf.math.equal(r_denom,0),dtype=tf.float32)

  #p=tf.math.divide_no_nan(d[...,0],(d[...,0]+d[...,1]))
  #r=tf.math.divide_no_nan(d[...,0],(d[...,0]+d[...,2]))
  #p+=p_nan
  #r+=r_nan

  # nan to 0
  p=tf.math.divide_no_nan(d[...,0],(d[...,0]+d[...,1]))
  r=tf.math.divide_no_nan(d[...,0],(d[...,0]+d[...,2]))

  return p,r #return precision, recall per class

def xyxy_to_xywh(batch):
  result=[[] for i in range(len(batch))]
  for i in range(len(batch)):
    if len(batch[i])>0:
      cur_image=batch[i][0]
      cur_image_w,cur_image_h=cur_image[...,2]-cur_image[...,0], cur_image[...,3]-cur_image[...,1]
      w,h=tf.reshape(cur_image_w, (len(cur_image_w),1)),tf.reshape(cur_image_h, (len(cur_image_h),1))
      result[i].append(tf.concat( (cur_image[...,:2],w,h,cur_image[...,4:6]),-1 ))
  return result
  

#parallel, coco, fixing.
def pr_per_conf_coco(filenames,filename_id,id_boxes,image_wh,img_path,batch_size,input_size,conf, metric_iou,box_iou):
  result=np.zeros((nc,3)) # TP,FP,TN
  
  def helper(step):
  #for i in range(step):
    batch,target=build_coco_data(filenames,filename_id,id_boxes,image_wh,step,batch_size,img_path,input_size,normalize=True)
    pred=model(batch)
    boxes=get_boxes(pred,input_size,grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=box_iou)
    val_boxes=xyxy_to_xywh(boxes)
    val_target=target_val(target, len(batch))
    
    rst=evaluate_tf(val_boxes,val_target,nc,metric_iou)
    
    return rst
  _=tf.map_fn(fn=helper,elems=tf.cast(tf.range(len(filenames)//batch_size+1),dtype=tf.int64) )
    #p,r=compute_pr_per_class(rst)
    #total_p+=p
    #total_r+=r
  
  return tf.reduce_sum(_,0)

#not parallel, coco
def pr_per_conf2_coco(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,conf, metric_iou,box_iou):
  
  result=np.zeros((nc,3)) # TP,FP,TN

  for step in range(len(filenames)//batch_size+1):
    batch,target=data.build_data_coco(filenames,filename_id,id_boxes,image_wh,step,batch_size,img_path,input_size,normalize=True)
    pred=model(batch)
    boxes=util.get_boxes(pred,input_size,grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=box_iou)
    val_boxes=xyxy_to_xywh(boxes)
    val_target=target_val(target, len(batch))
    
    rst=evaluate_tf(val_boxes,val_target,metric_iou)
    
    result+=rst
    #p,r=compute_pr_per_class(rst)
    #total_p+=p
    #total_r+=r
  return result

#parallel, coco
def pr_per_conf2_coco_pa(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,conf, metric_iou,box_iou):
  
  result=np.zeros((nc,3)) # TP,FP,TN

  #for step in range(len(filenames)//batch_size+1):
  def helper(step):
    batch,target=data.build_data_coco(filenames,filename_id,id_boxes,image_wh,step,batch_size,img_path,input_size,normalize=True)
    pred=model(batch)
    boxes=util.get_boxes(pred,input_size,grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=box_iou)
    val_boxes=xyxy_to_xywh(boxes)
    val_target=target_val(target, len(batch))
    
    rst=evaluate_tf(val_boxes,val_target,metric_iou)
    return rst

  _=tf.map_fn(fn=helper,elems=tf.cast(tf.range(len(filenames)//batch_size+1),dtype=tf.int64))
    #result+=rst
    #p,r=compute_pr_per_class(rst)
    #total_p+=p
    #total_r+=r
  return tf.reduce_sum(_,0)


#not parallel, not coco
def pr_per_conf(filenames,img_path,label_path,batch_size,input_size,conf, metric_iou,box_iou):
  result=np.zeros((nc,3)) # TP,FP,TN
  
  def helper(step):
  #for i in range(step):
    print(step)
    batch,target=build_data(filenames,img_path,label_path,step,batch_size,input_size,normalize=True)
    pred=model(batch)
    boxes=get_boxes(pred,input_size,grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=box_iou)
    val_boxes=xyxy_to_xywh(boxes)
    val_target=target_val(target, len(batch))
    
    rst=evaluate_tf(val_boxes,val_target,nc,metric_iou)
    
    return rst
  _=tf.map_fn(fn=helper,elems=tf.cast(tf.range(len(filenames)//batch_size+1),dtype=tf.int64) )
    #p,r=compute_pr_per_class(rst)
    #total_p+=p
    #total_r+=r
  
  return tf.reduce_sum(_,0)

#not parallel, not coco
def pr_per_conf2(filenames,img_path,label_path,batch_size,input_size,conf, metric_iou,box_iou):
  result=np.zeros((nc,3)) # TP,FP,TN
  for i in range(len(filenames)//batch_size+1):
    batch,target=build_data(filenames,img_path,label_path,i,batch_size,input_size,normalize=True)
    pred=model(batch)
    boxes=get_boxes(pred,input_size,grids,len(batch),anchors,nl,conf_threshold=conf,iou_threshold=box_iou)
    val_boxes=xyxy_to_xywh(boxes)
    val_target=target_val(target, len(batch))
    
    rst=evaluate_tf2(val_boxes,val_target,nc,metric_iou)
    
    result+=rst
    #p,r=compute_pr_per_class(rst)
    #total_p+=p
    #total_r+=r
  return result




# box_iou in argument is for get_boxes function
def validate_pr_coco(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,box_iou=0.5):
  metric_iou=tf.constant([0.5,0.95])
  
  #pr_per_class=np.zeros((len(metric_iou),2,nc))
  #map=np.zeros(len(metric_iou))
  
  def helper(iou):
    pr,ap=pr_list_coco(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,iou,box_iou)
    return ap
  _=tf.map_fn(fn=helper,elems=metric_iou)
  
  return _  # metrics for each iou, (2,) for now


# For a given iou, 
# compute p,r of each class. Then, get mean p, r per class across conf(-.01 ~ 9.9) and mean of those means.
# box_iou in argument is for get_boxes function
def pr_list_coco(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,metric_iou,box_iou):
  #result=np.zeros((11,2)) # ap, ar of each conf
  def helper(idx):
    conf=-0.01+0.1*idx
    #rr=pr_per_conf2_coco(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,conf, metric_iou,0.5)
    rr=pr_per_conf2_coco_pa(filenames,filename_id,id_boxes,image_wh,model,img_path,batch_size,input_size,conf, metric_iou,0.5)
    p,r=compute_pr_per_class(rr)
    return tf.stack((p,r))
  
  _=tf.map_fn(fn=helper,elems=tf.cast(tf.range(11),dtype=tf.float32))
  
  pr=tf.concat( (tf.reduce_mean(_,-1),[[1,0]]),0 )
  sorted_r_index=tf.argsort(pr[...,1])
  pr=tf.gather(pr,sorted_r_index)
  p,r=pr[...,0],pr[...,1]
  ap=tf.reduce_sum((r[:-1]-r[1:])*p[:-1])

  mean_pr=tf.reduce_mean(_,0) # mean p,r per class [2, number of classes]
  #mean_pr_conf=tf.reduce_mean(mean_pr,1) # mean of mean p,r per class

  return mean_pr, ap # [2, number of classes], [1]
'''
def pr_list(filenames,filename_id,id_boxes,image_wh,img_path,batch_size,input_size,metric_iou,box_iou):
  result=np.zeros((11,2)) # ap, ar of each conf
  conf=-0.01
  for i in range(11):  
    rr=pr_per_conf2_coco(filenames,filename_id,id_boxes,image_wh,img_path,batch_size,input_size,conf, 0.5,0.5)
    p,r=compute_pr_per_class(rr)
    ap,ar=np.mean(p),np.mean(r) 
    result[i][0]=ap
    result[i][1]=ar
    print(ap,ar)
    conf+=0.1
  return result
'''
def pr_list2(filenames,img_path,label_path,batch_size,input_size,metric_iou,box_iou):
  result=np.zeros((11,2)) # ap, ar of each conf
  conf=-0.01
  for i in range(11):  
    rr=pr_per_conf2(filenames,img_path,label_path,batch_size,input_size,conf,metric_iou,box_iou)
    p,r=compute_pr_per_class(rr)
    ap,ar=np.mean(p),np.mean(r) 
    result[i][0]=ap
    result[i][1]=ar
    print(ap,ar)
    conf+=0.1
  return result
