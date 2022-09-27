import argparse
import tensorflow as tf
from data import models, util, data, metric
import numpy as np
import os
import time
import tqdm
import math

def step(x,y,bs,bs_,model,grids,img_size,anchors,hyp):
  with tf.GradientTape() as tape:
    pred=model(x)
    loss,loss_tuple=util.compute_loss2(pred,y,anchors,bs,bs_,grids,img_size,hyp)
  grads=tape.gradient(loss,model.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

  
  

  return loss,loss_tuple




#train_dict : (filenames, filename_id, id_boxes, image_wh)
#val_dict : (filenames, filename_id, id_boxes, image_wh)
def train_coco(epochs, batch_size,train_dict,val_dict,img_path,val_img_path,label_path,input_size,model,grids,img_size,anchors,hyp,conf,iou,save_path):
  #print('{:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('epoch','total','objectness', 'confidnece', 'class'))

  filenames,filename_id,id_boxes,image_wh=train_dict[0],train_dict[1],train_dict[2],train_dict[3]
  filenames_val,filename_id_val,id_boxes_val,image_wh_val=val_dict[0],val_dict[1],val_dict[2],val_dict[3]

  log = open("log.txt", "w")
  
  print('\n\n')
  for j in range(epochs):
    loss_arr=np.array([0.0,0.0,0.0,0.0])
    
    train_start_time=time.time()
    train_step=math.ceil(len(filenames)/batch_size)
    tmp_str='{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('epoch','step','time','total','objectness', 'confidnece', 'class')
    log.write(tmp_str+'\n')
    #print('{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('epoch','time','total','objectness', 'confidnece', 'class'))
    print(tmp_str)
    for i in range(train_step):#len(step_size)
      batch,target=data.build_data_coco(filenames, filename_id,id_boxes,image_wh,i,batch_size,img_path,input_size, normalize=True)
      loss,lt=step(batch,target,len(batch),len(batch),model,grids,img_size,anchors,hyp)
      loss_arr*=i
      loss_arr+=[loss[0].numpy(),lt[0][0].numpy(),lt[1][0].numpy(),lt[2][0].numpy()]
      loss_arr/=(i+1)
      train_now=time.time()-train_start_time
      tmp_str='{:>12}  {:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format(str(j)+'/'+str(epochs-1),str(i)+'/'+str(train_step-1),train_now,loss_arr[0],loss_arr[1], loss_arr[2], loss_arr[3])  
      print(tmp_str,end='\r')
    print(tmp_str)
    log.write(tmp_str+'\n')
    
  
    #loss_arr/=(len(filenames)//batch_size+1)
    #train_end_time=time.time()
    #train_time=train_end_time-train_start_time
    #train_time=time_now
    
    #tmp_str='{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format(str(j)+'/'+str(epochs-1),train_time,loss_arr[0],loss_arr[1], loss_arr[2], loss_arr[3])
    #log.write(tmp_str)
    #print(tmp_str)
    #print('{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format(str(j)+'/'+str(epochs-1),train_time,loss_arr[0],loss_arr[1], loss_arr[2], loss_arr[3]))

    
    val_loss_arr=np.array([0.0,0.0,0.0,0.0])
    val_start_time=time.time()
    val_step=math.ceil(len(filenames_val)/batch_size)
    tmp_str='{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('valid','step','time','total','objectness', 'confidnece', 'class')
    log.write(tmp_str+'\n')
    print(tmp_str)
    for i in range(val_step):
      batch,target=data.build_data_coco(filenames_val, filename_id_val,id_boxes_val,image_wh_val,i,batch_size,val_img_path,input_size, normalize=True)
      pred=model(batch)
      val_loss,val_lt=util.compute_loss2(pred,target,anchors,len(batch),len(batch),grids,img_size,hyp)
      val_loss_arr*=i
      val_loss_arr+=[val_loss[0].numpy(),val_lt[0][0].numpy(),val_lt[1][0].numpy(),val_lt[2][0].numpy()]
      val_loss_arr/=(i+1)
      val_now=time.time()-val_start_time
      tmp_str='{:>12}  {:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format('',str(i)+'/'+str(val_step-1),val_now,val_loss_arr[0],val_loss_arr[1], val_loss_arr[2], val_loss_arr[3])  
      print(tmp_str,end='\r')
    print(tmp_str)
    log.write(tmp_str+'\n')
    #val_loss_arr/=(len(filenames_val)//batch_size+1)
    
    
    #val_end_time=time.time()
    #val_time=val_end_time-val_start_time
    #tmp_str='{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('valid','time','total','objectness', 'confidnece', 'class')
    #log.write(tmp_str+'\n')
    #print(tmp_str)
    #print('{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('valid','time','total','objectness', 'confidnece', 'class'))
    #tmp_str='{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format('',val_time,val_loss_arr[0],val_loss_arr[1], val_loss_arr[2], val_loss_arr[3])
    #log.write(tmp_str+'\n')
    #print(tmp_str)
    #print('{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format('',val_time,val_loss_arr[0],val_loss_arr[1], val_loss_arr[2], val_loss_arr[3]))
    
    '''
    if j%100==99 or j==epochs-1:
      maps_start_time=time.time()
      maps=metric.validate_pr_coco(filenames_val, filename_id_val,id_boxes_val,image_wh_val,model,val_img_path,batch_size,input_size,box_iou=0.5)
      maps_end_time=time.time()
      maps_time=maps_end_time-maps_start_time
      tmp_str='{:>12}  {:>12}  {:>12}  {:>12}'.format('map','time','map@0.5','map@0.95')
      log.write(tmp_str+'\n')
      print(tmp_str)
      #print('{:>12}  {:>12}  {:>12}  {:>12}'.format('map','time','map@0.5','map@0.95'))
      tmp_str='{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format('',maps_time,maps[0],maps[1])
      log.write(tmp_str+'\n')
      print(tmp_str)
      #print('{:>12}  {:>12.8f}  {:>12.8f}  {:>12.8f}'.format('',maps_time,maps[0],maps[1]))
      model.save_weights(save_path+'weights.h5')
    '''
    model.save_weights(save_path+'weights'+"{:02d}".format(j)+'.h5')
    log.write(''+'\n')
    print('')

  log.close()
    

def args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=640, help='input size')
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--val_img_path', type=str)
    parser.add_argument('--label_path', type=str, help='file path if coco, folder path that contains label files otherwise')
    parser.add_argument('--val_label_path', type=str, help='folder path that contains label files')
    parser.add_argument('--weights', type=str, help='file path to the weights to be loaded')
    parser.add_argument('--width', type=float, default=1.0, help='network width')
    parser.add_argument('--depth', type=float, default=1.0, help='network depth')
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--save_weights_path', type=str, default='', help='file path to save trained weights')
    parser.add_argument('--iou',type=float,default=0.5, help='iou to be used for validation')
    parser.add_argument('--conf',type=float,default=0.2, help='confidence to be used for validation')
    parser.add_argument('--nc',type=int, default=90, help='number of classes')
    parser.add_argument('--lr',type=float, default=0.001, help= 'learning rate')
    args=parser.parse_args()

    return args


    
def main():
    arg=args()
    
    batch_size=arg.batch
    img_path=arg.img_path
    val_img_path=arg.val_img_path
    label_path=arg.label_path
    val_label_path=arg.val_label_path
    width=arg.width
    depth=arg.depth
    learning_rate=arg.learning_rate
    img_size=arg.img_size
    input_size=arg.img_size
    epochs=arg.epochs
    nc=arg.nc
    conf=arg.conf
    iou=arg.iou
    lr=arg.lr
    anchors=np.array([[[10.0,13.0], [16.0,30.0], [33.0,23.0]]  ,# P3/8
                        [[30.0,61.0], [62.0,45.0], [59.0,119.0]]  ,# P4/16
                        [[116.0,90.0], [156.0,198.0], [373.0,326.0]] ],dtype='float32') # P5/32
    
    hyp_={'balance':np.array([4.0,1.0,0.4]),
          'box': np.array([1, 0.02, 0.2]),  # box loss gain\\ 
          'cls': np.array([1, 0.2, 4.0]),  # cls loss gain\\
          'cls_pw': np.array([1, 0.5, 2.0]),  # cls BCELoss positive_weight\\
          'obj': np.array([1, 0.2, 4.0]),  # obj loss gain (scale with pixels)\\
          'obj_pw': np.array([1, 0.5, 2.0]), } # obj BCELoss positive_weight

    hyp_['obj']*=(img_size/640)**2
    anchors*=img_size/640

    model,grids=models.get_model(img_size,nc,width,depth,lr)
    metric.set_args(nc,grids,anchors)
    if arg.weights:
        model.load_weights(arg.weights)

    
    filenames=os.listdir(img_path)

    if arg.coco:
      filenames,filename_id,id_boxes,image_wh=data.build_coco(img_path,label_path)
      filenames_val,filename_id_val,id_boxes_val,image_wh_val=data.build_coco(val_img_path,val_label_path)
      train_dict=(filenames,filename_id,id_boxes,image_wh)
      val_dict=(filenames_val,filename_id_val,id_boxes_val,image_wh_val)
      
      train_coco(epochs, batch_size,train_dict,val_dict,img_path,val_img_path,label_path,input_size,model,grids,img_size,anchors,hyp_,conf,iou,arg.save_weights_path)
      #train_coco(epochs,batch_size,filenames,img_path,label_path,model,grids,img_size,anchors,hyp_,arg.conf,arg.iou)

    model.save_weights(arg.save_weights_path+'weights.h5')

    

if __name__=="__main__":
    
    main()

