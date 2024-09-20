# Script to convert the Doclaynet COCO Format to COCO128 Format used by the yolov5
import json 
import shutil
import os 

error_log = open("error.txt", 'a+')
def read_json(json_path:str): 
    with open(json_path, 'r') as fp: 
        json_data= json.load(fp)
    return json_data 
        
        
def normalize_bbox(ann, img_w, img_h):
    x, y, w, h = ann['bbox']  # x, y are top-left corner coordinates
    # Calculate center of the bounding box
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize the center coordinates and dimensions, ensuring they are strictly less than 1
    x_center_normalized = min(x_center / img_w, 0.9999)
    y_center_normalized = min(y_center / img_h, 0.9999)
    w_normalized = min(w / img_w, 0.9999)
    h_normalized = min(h / img_h, 0.9999)


    return [x_center_normalized, y_center_normalized, w_normalized, h_normalized]


def create_dataset(json_path, IMG_DIR, OUTPUT_LABELS_DIR, OUTPUT_IMAGES_DIR): 
    json_data = read_json(json_path)
    
    class_mapping = {i+1:i  for i in range(11)}  
    # Preprocess annotations into a dictionary keyed by image_id       
    annotations_by_image = {}
    for ann in json_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = [] 
            
        annotations_by_image[image_id].append(ann)
    
        
    for idx, image in enumerate(json_data['images']): 
        # Normalizing the Coordinates 
        try: 
            anns = annotations_by_image[image['id']] # Get all the annotations  of the image 
        except KeyError: 
            # KeyError occurs when there is no Key which means there is no annotations for that image so skip those s
            error_log.write(str(image['id']))
            error_log.write('\n')
            continue
        file_name = image['file_name']
        img_w = image['width']
        img_h = image['height']
        print(f"{len(json_data['images'])-idx} images remaining")
        with open(f"{OUTPUT_LABELS_DIR}/{file_name.split('.png')[0]}.txt",'w') as fp: 
            for ann in anns: 
                cat_id = ann['category_id']
                new_cat_id = class_mapping[cat_id]
                bbox_yolo = normalize_bbox(ann, img_w, img_h)
                fp.write(f"{new_cat_id} {bbox_yolo[0]:0.6f} {bbox_yolo[1]:0.6f} {bbox_yolo[2]:0.6f} {bbox_yolo[3]:0.6f}") 
                fp.write("\n")
        shutil.copy(os.path.join(IMG_DIR,f"{file_name}"), OUTPUT_IMAGES_DIR )
        
    
if __name__ == "__main__": 
    data_type = "val"
    doclaynet_dir = "data_doclaynet" # Dir containing doclaynet documents 
    JSON_DIR = doclaynet_dir+ f"/COCO/{data_type}.json"
    IMG_DIR = os.path.join(doclaynet_dir, "PNG")
    OUTPUT_DIR =f"COCO"
    

    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    os.makedirs(OUTPUT_DIR+ "/images",exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/images/"+ f"{data_type}2017",exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/labels",exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/labels/"+ f"{data_type}2017",exist_ok=True)
    OUTPUT_LABELS_DIR = OUTPUT_DIR+"/labels/"+ f"{data_type}2017"
    OUTPUT_IMAGES_DIR = OUTPUT_DIR+"/images/"+ f"{data_type}2017"
    create_dataset(JSON_DIR, IMG_DIR, OUTPUT_LABELS_DIR, OUTPUT_IMAGES_DIR)
    print("All Conversion Completed")
    error_log.close()