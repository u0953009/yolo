# Yolo implementation
## Reference 
https://github.com/ultralytics/yolov5

## Train
 
 
 ```sh
 - This implementation was built based on COCO dataset format 
  python train.py --coco 
                  --epochs                   # number of epochs
                  --batch                    # number of batch
                  --img_size                 # input size of network
                  --img_path                 # folder path to the images for train
                  --val_img_path             # folder path to the images for validation
                  --label_path               # COCO data label path for train
                  --val_label_path           # COCO data label path for validation
                  --width                    # network width, default=1.0
                  --depth                    # network depth, default=1.0 
                  --weights                  # file path to the weights file, optional
- If you want to load weights, width and depth have to match to the shape of the weights
- example
 python train.py --coco --epochs 1 --batch 20 --img_size 416 --img_path demo/val2017_/ --val_img_path demo/val2017_/ --label_path demo/label.json --val_label_path demo/label.json --width 0.33 --depth 0.22
 ```
 
 ## Inference
 
 ```sh
  
  python predict.py --input_size               # input size of network
                    --img_path                 # folder path to the images for inference
                    --result_path              # folder path to save the result
                    --weights                  # file path to the weights
                    --width                    # network width of the saved weights
                    --depth                    # network depth of the saved weights
- example
python predict.py --input_size 416 --img_path demo/test/ --result_path demo/result/ --weights demo/weights00.h5 --width 0.33 --depth 0.22
 ```

