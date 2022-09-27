

import numpy as np
import tensorflow as tf
import math

na=3
nl=3



def merge(arr1,arr2,idx):
  end1,end2=len(arr1)-1,len(arr2)-1
  idx1,idx2=0,0
  cur_idx=0
  result=np.empty((0,arr1.shape[1]))
  

  while idx1<=end1 and idx2<=end2:
    
    if arr1[idx1][idx]<=arr2[idx2][idx]:
      result=np.append(result,[arr1[idx1]],axis=0)
      idx1+=1
    else:
      result=np.append(result,[arr2[idx2]],axis=0)
      idx2+=1

  if idx1<=end1:
    result=np.append(result,arr1[idx1:],axis=0)
  if idx2<=end2:
    result=np.append(result,arr2[idx2:],axis=0)

  return result



def merge_sort(arr,idx):
  if len(arr)==1:
    return arr
  if len(arr)==2:
    if arr[0][idx]<=arr[1][idx]:
      return arr
    else:
      return np.array([arr[1],arr[0]])

  half1=merge_sort(arr[:int(len(arr)/2)],idx)
  half2=merge_sort(arr[int(len(arr)/2):],idx)

  cc=merge(half1,half2,idx)

  return cc

def np_sort(arr,cur,base):

  if cur==base:
    return merge_sort(arr,cur)
  dictt={}
  result=np.empty((0,arr.shape[1]))
  arrlen=len(arr)

  for i in range(arrlen):
    tmp=arr[i][cur]
    if tmp not in dictt:
      dictt[tmp]=np.empty((0,arr.shape[1]))
    dictt[tmp]=np.append(dictt[tmp],[arr[i]],axis=0)
  
  
  
  keysort=np.array([])
  for k in dictt.keys():
    keysort=np.append(keysort,[k])
  keysort.sort()  
  

  for i in range(len(keysort)):
    result=np.append(result,np_sort(dictt[keysort[i]],cur+1,base),axis=0)

  return result

  # compare rows by column from v1 to v
def remove_redun(arr,v1,v):
  idx=len(arr)-1
  v+=1
  to_be_del=[]
  while idx>0:
    if np.all( (arr[idx][v1:v]==arr[idx-1][v1:v])==True):
      to_be_del.append(idx)
    idx-=1
  
  if len(to_be_del)==0:
    return arr
  else:
    return np.delete(arr,np.array(to_be_del),axis=0)
# box1 : ground truth, box2 : prediction
def box_iou(box1,box2,  xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
  b1x,b1y,b1w,b1h=tf.split(box1,[1,1,1,1],axis=-1)
  b2x,b2y,b2w,b2h=tf.split(box2,[1,1,1,1],axis=-1)

  b1x1,b1x2,b1y1,b1y2= b1x,b1x+b1w,b1y,b1y-b1h
  b2x1,b2x2,b2y1,b2y2= b2x,b2x+b2w,b2y,b2y-b2h
  
  intersect=tf.math.maximum(tf.math.minimum(b1x2,b2x2)-tf.math.maximum(b1x1,b2x1),0)*tf.math.maximum(tf.math.minimum(b1y1,b2y1)-tf.math.maximum(b1y2,b2y2),0)
  union= b1w*b1h+b2w*b2h-intersect+eps

  iou=intersect/union
  
  if CIoU:
    b1cx=(b1x1+b1x2)/2
    b1cy=(b1y1+b1y2)/2
    b2cx=(b2x1+b2x2)/2
    b2cy=(b2y1+b2y2)/2
    dist=((b1cx-b2cx)**2+(b1cy-b2cy)**2)/( (tf.math.maximum(b1x2,b2x2)-tf.math.minimum(b1x1,b2x1))**2 + (tf.math.maximum(b1y1,b2y1)-tf.math.minimum(b1y2,b2y2))**2)
    v=4/ (math.pi**2 )*(tf.math.atan(b1w/b1h)-tf.math.atan(b2w/b2h))**2
    alpha = v / (v - iou + (1 + eps))
    iou-=(dist+alpha*v)
  return iou
  
def extend_targets(target,anchors,na,nl,grids,img_size):

  
  nt=target.shape[0]
  ne=target.shape[1]

  grid_scale=(grids/img_size)
  
  grid_scale=np.reshape(grid_scale,(nl,1) )

  scaled_anchors=np.multiply(anchors,grid_scale[:,None])

  
  target1=np.reshape(np.repeat(target,3,axis=0), (nt,3,ne)  )  
  scale=np.tile(np.ones(ne) ,(na,1))

  scale[...,2:]=np.multiply(scale[...,2:],grids[:,None])
  target=np.multiply(target[:,None],scale)
  target=np.stack([target[:,0,:],target[:,1,:],target[:,2,:] ])

    
  an=np.arange(na)
  
  target=np.repeat(target,na, axis=-2)
  an=np.reshape(np.tile(an,(nl,nt)), (nl,nt*na,1) ) 
  

  target=np.concatenate((target,an),axis=-1)  # number in each batch, class, x,y,w,h,anchor number in each layer


  target_wh=target[...,4:6]
 
  anchor_target=np.reshape(np.repeat(scaled_anchors, nt,axis=0 ),(nl,nt*na,2) )

  anchor_hyp=4
  r= target_wh/anchor_target
  
  r=np.maximum(r,1/r)
  r=np.maximum(r[...,0],r[...,1]) < anchor_hyp


  off=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])
  g=0.5

  result=[]

  for i in range(nl):

    targetl=target[i][r[i]]
    gxy=targetl[...,2:4]
    gxi=grids[i]-gxy

    j,k=np.transpose(((gxy%1<g) & (gxy>1)) )
    l,m=np.transpose(((gxi%1<g) & (gxi>1)) )
    j=np.stack( (np.ones_like(j),j,k,l,m) )
    targetl=np.tile(targetl,(5,1,1))[j]
    offsets=(np.zeros_like(gxy)[None]+off[:,None])[j] 
   
    xy=targetl[...,2:4]
    
    ij=np.floor(xy-offsets)
    xy=xy-ij

    # class,number in batch, i,j,a,x offset,y offset,w,h
    targetl=np.concatenate( (targetl[...,1][:,None],targetl[...,0][:,None], ij,targetl[...,6][:,None],xy,targetl[...,4:6]),-1) 
    
    
    
    targetl=np_sort(targetl,1,4)
    
    targetl=remove_redun(targetl,1,4)

    
    #number in batch, class, i,j,a,x offset,y offset,w,h
    targetl=np.concatenate( (targetl[...,1][:,None],targetl[...,0][:,None],targetl[...,2:]) ,-1) 
    
    
    result.append(targetl)
    


  return result, scaled_anchors
    


def extract_indices(targets,anchors):
  tcls,tbox,indices,anch=[],[],[],[]
  for i in range(len(targets)):
    
    targetl=targets[i]
    a=targetl[...,4]
    b=targetl[...,0]
    c=targetl[...,1]
    ij=targetl[...,2:4]
    xy=targetl[...,5:7]
    wh=targetl[...,7:]
    
    ti,tj=np.transpose(ij)
    indices.append((b.astype(int),a.astype(int),tj.astype(int),ti.astype(int)))
    tbox.append(np.concatenate((np.float32(xy),np.float32(wh)),-1))
    anch.append(np.take(anchors[i],a.astype(int),axis=0))
    tcls.append(c.astype(int))
    
  return tcls,tbox,indices,anch





#y: number in batch, class, x,y,w,h
def compute_loss2(y_pred,y,anchors,bs,bs_,grids,img_size,hyp):
  lbox=tf.zeros(1)
  lobj=tf.zeros(1)
  lcls=tf.zeros(1)
  
  nc=y_pred[0].shape[4]-5
  
  #tcls,tbox,indices,anch=extract_indices(extend_targets(y,aanchors,3,3,np.array([52,26,13])),anchors)
  etargets,scaled_anchors=extend_targets(y,anchors,na,nl,grids,img_size)
  tcls,tbox,indices,anch=extract_indices(etargets,scaled_anchors)
  
  
  for i in range(len(y_pred)):
    #print(len(indices[i][0]))
    #print(i)
    

    if len(indices[i][0])>0:
      box_indices=tf.transpose(tf.stack( (indices[i][0],indices[i][1],indices[i][2],indices[i][3]) ))
      
      
      p_box=tf.gather_nd(y_pred[i],box_indices)[...,:4]
      pxy,pwh=tf.split(p_box,[2,2],-1)
      pxy=tf.sigmoid(pxy)*2-0.5
      pwh= (tf.sigmoid(pwh)*2)**2*anch[i]
      p_box=tf.concat((pxy,pwh),-1)
      
      t_box=tbox[i]

      iou=box_iou(t_box,p_box,CIoU=True)
      

      lbox+= tf.math.reduce_mean(1-iou)*hyp['box'][i]
      

      #iou=tf.stop_gradient(tf.clip_by_value(iou,0))
      i#ou=tf.clip_by_value(iou,0, 100)
      
      
      # this one works
      tobj=tf.sparse.SparseTensor(tf.cast(box_indices,dtype=tf.int64),tf.ones(box_indices.shape[0]), (bs, 3,y_pred[i].shape[2],y_pred[i].shape[3]))
      
      #tobj=tf.sparse.SparseTensor(tf.cast(box_indices,dtype=tf.int64),tf.squeeze(iou), (bs, 3,y_pred[i].shape[2],y_pred[i].shape[3]))
      #tobj=tf.sparse.SparseTensor(tf.cast(box_indices,dtype=tf.int64),tf.squeeze(iou), (bs, na,y_pred[i].shape[2],y_pred[i].shape[3]))
      tobj=tf.sparse.reorder(tobj)    
      
      
      tobj=tf.sparse.to_dense(tobj)
      

      objloss=tf.nn.sigmoid_cross_entropy_with_logits(tobj,y_pred[i][...,4])
      
      
      lobj+=tf.reduce_mean(objloss)*hyp['balance'][i]*hyp['obj'][i]

      # class loss
      p_cls=tf.gather_nd(y_pred[i],box_indices )[...,5:]
      t_cls=np.zeros( (len(tcls[i]),nc ))
      t_cls[np.arange(len(tcls[i])),tcls[i]]=1
      t_cls=tf.constant(t_cls, dtype=tf.float32)

      clsloss=tf.nn.sigmoid_cross_entropy_with_logits(t_cls,p_cls)
      #clsloss=tf.nn.softmax_cross_entropy_with_logits(t_cls,p_cls)

      #print('class loss : ', tf.reduce_mean(clsloss))
      lcls+=tf.reduce_mean(clsloss)*hyp['cls'][i]
    else:
      lobj+= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros(y_pred[i][...,4].shape),y_pred[i][...,4]))*hyp['obj'][i]
  
  #lbox *= hyp['box']
  #lobj *= hyp['obj']
  #lcls *= hyp['cls']    
  #return (lbox+lobj+lcls)*bs_, (lbox,lobj,lcls)
  return (lbox+lobj+lcls), (lbox,lobj,lcls)


def get_boxes(y_pred,input_size, grids,bs,anchors,nl,conf_threshold=0.2,iou_threshold=0.5):
  batch_boxes=[[] for i in range(bs)]
  
  
  for i in range(nl):
    conf=tf.sigmoid(y_pred[i][...,4])
    score_indices=tf.where(tf.math.greater(conf,conf_threshold))

    #print('total indices :', score_indices.shape)
    
    
    if len(score_indices)>0:
      boxes_xywh=tf.gather_nd(y_pred[i],score_indices)
      xy=tf.sigmoid(boxes_xywh[...,:2])*2-0.5
      wh=(tf.sigmoid(boxes_xywh[...,2:4])*2)**2*anchors[[i],[score_indices[...,1]]]*grids[i]/input_size
      cls=tf.argmax(tf.sigmoid(boxes_xywh[...,5:]),-1)
      
      conf=tf.sigmoid(boxes_xywh[...,4])

      
      x,y=xy[...,0],xy[...,1]
      ii,jj=tf.cast(score_indices[...,3], dtype=tf.float32 ), tf.cast(score_indices[...,2],dtype=tf.float32)
      
      x1,y1= (ii+x) , (jj+y)
      w,h=wh[...,0],wh[...,1]
      x2,y2= x1+w, y1+h
      x2,y2= tf.reshape(x2,(x2.shape[1],)), tf.reshape(y2,(y2.shape[1],))
      x1,y1,x2,y2=x1/grids[i],y1/grids[i],x2/grids[i],y2/grids[i]
      
      boxes=tf.concat( (tf.transpose(tf.stack((x1,y1,x2,y2,conf))), boxes_xywh[...,5:] ),-1 )       # x1,y1,x2,y2,confidence, ....classes....
      
      
      # Seperate boxes into each image
      def add_boxes(idx):
        if tf.gather_nd( boxes  ,tf.where(tf.math.equal(score_indices[...,0],idx))).shape[0]>0:
          if len(batch_boxes[idx]):
            batch_boxes[idx][0]=tf.concat((batch_boxes[idx][0],tf.gather_nd( boxes  ,tf.where(tf.math.equal(score_indices[...,0],idx)))),0)
          else:
            batch_boxes[idx].append(tf.gather_nd( boxes  ,tf.where(tf.math.equal(score_indices[...,0],idx))))
        return 1

      _=tf.map_fn(fn=add_boxes, elems=tf.cast(tf.range(bs),dtype=tf.int64) )
      # Non Maxima Suppression
      def nms(idx):
        if len(batch_boxes[idx])>0:
          nms_indices=tf.image.non_max_suppression(batch_boxes[idx][0][...,:4],batch_boxes[idx][0][...,4],150,iou_threshold)      
          batch_boxes[idx][0]=tf.gather_nd(batch_boxes[idx][0],nms_indices[...,None],batch_dims=0)
        return 1
      _=tf.map_fn(fn=nms, elems=tf.cast(tf.range(bs),dtype=tf.int64) )
  
    #print('i end\n')
    
  return batch_boxes



# no use
def per_batch(boxes, label,nc,iou_threshold):

  def per_img(idx):
    t,p=label[idx], boxes[idx]
    rst=np.zeros(nc,3) # TP, FP, TN

    
    for i in range(len(t)): # iterate label(true boxes)
      cur_t=t[i]
      no_match=1 # if this true box doesn't find a predicted box
      best_iou=0
      best_box=-1

      
      for j in range(len(p)): # iterate predicted box per true box
        cur_p=p[j]
        
        if cur_p[5]==cur_t[0]:
          iou=box_iou(cur_t,cur_p[...,:4])
        if iou>iou_teshold:
          no_match=0
          if iou>best_iou:
            best_iou=iou
            best_box=j

      if no_match:
        arr[cur_t[0]][2]+=1
      else:
        arr[cur_t[0]][0]+=1
        np.remove(p,best_box)

    # reamining predicted boxes
    for i in range(len(p)):
      arr[p[j][5]][1]+=1

    return rst
  _=tf.map_fn(fn=per_img,elems=tf.cast(tf.range(len(boxes)),dtype=tf.int64))

  return tf.reduce_sum(_,0)



  
  
            
          
      
        
        
        





















  
  


