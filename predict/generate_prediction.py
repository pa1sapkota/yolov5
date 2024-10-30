from .post_process import postprocess_dit 
from .visualize import visualize_prediction
import json 
import os 
import cv2 


    

def read_json(file_path:str): 
    with open(file_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data


JSON_DIR = "/home/ubuntu/yolov5/outputs/oct25_finance"
IMG_DIR = "/home/ubuntu/yolov5/downloads/datasets/data_doclaynet"
exp_name = "finance_october25_post_process"
OUTPUT_DIR = f"outputs/{exp_name}"
JSON_SAVE_DIR = f"{OUTPUT_DIR}/post_process_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_SAVE_DIR, exist_ok=True)
JSON_FILES = os.listdir(JSON_DIR)


for file in JSON_FILES: 
    img_name = file.split(".json")[0]
    full_img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(full_img_path)
    full_file_path = os.path.join(JSON_DIR, file)
    import pdb; pdb.set_trace()
    raw_json = read_json(full_file_path)
    post_process_json = postprocess_dit(raw_json['jsonData']['result'][0])  
    # Save the post_process json 
    with open(f"{JSON_SAVE_DIR}/{img_name}.json",'w') as fp: 
        json.dump(post_process_json, fp, indent=4)
    
    post_process_img = visualize_prediction(post_process_json, img, raw=False)
    cv2.imwrite(f"{OUTPUT_DIR}/{img_name}", post_process_img)
    print(f"Image: {img_name} processed successfully")
    
