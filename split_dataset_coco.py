import json 
import random 
from collections import defaultdict

SEED_NUMBER = 444
SAVE_DIR = "/mnt/c/Users/FM-PC-LT-356/Documents/yolov5/financial_data/COCO"
random.seed(SEED_NUMBER)

def read_json(json_path:str): 
    with open(json_path, 'r') as fp: 
        json_data= json.load(fp)
    return json_data 
        

json_path = "/mnt/c/Users/FM-PC-LT-356/Documents/yolov5/financial_data/post_process/train.json"
data_json = read_json(json_path)

images = data_json['images'] 


train_test_split_ratio = 0.8 

random.shuffle(images)
split_index = int(len(images) * train_test_split_ratio)
print(split_index)
train_images = images[:split_index]
val_images = images[split_index:] 
annotation_train = [] 
annotation_val = [] 
categories = [category for category in data_json['categories']][:12]
train_ids = [image['id'] for image in train_images] # get train ids 
val_ids = [image['id'] for image in val_images] # get val ids   
print(len(val_ids), len(train_ids))
# breakpoint()
for annotation in data_json['annotations']: 
    if annotation['image_id'] in train_ids: 
        annotation_train.append(annotation) 
    elif annotation['image_id'] in val_ids: 
        annotation_val.append(annotation) 
    else : 
        print(f"Error Fix Some Issues")   
json_train = {
    "categories": categories, 
    "images": train_images, 
    "annotations": annotation_train
}
        
json_val = {
    "categories": categories, 
    "images": val_images,  
    "annotations": annotation_val
}
with open(f"{SAVE_DIR}/train.json",'w' ) as fp: 
    json.dump(json_train, fp, indent=4)

with open(f"{SAVE_DIR}/val.json",'w' ) as fp: 
    json.dump(json_val, fp, indent=4)