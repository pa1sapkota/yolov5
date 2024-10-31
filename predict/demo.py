
import os
import subprocess
import torch
import yaml 
import json 
from predict.post_process import postprocess_dit
from predict.visualize import visualize_prediction
import cv2 
from PIL import Image 


with open("predict/configs.yaml",'r') as fp: 
    config = yaml.safe_load(fp) 

    
def read_json(file_path:str): 
    with open(file_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data


# Load all the configs from the config file 
input_dir = config['input_images_dir']  
output_dir = config['output_dir']  
os.makedirs(output_dir, exist_ok=True)
weights_path = config['model_checkpoint'] 
conf = config['confidence']
experiment_name = config['experiment_name']
IMG_TYPE = ('.jpg', '.png') # Endswith take tuple or string as arg
class_dict = config['class_dict']


IMAGES = os.listdir(input_dir) 
IMAGES = sorted([img for img in IMAGES if img.endswith(IMG_TYPE)])  
LABELS_DIR = os.path.join(output_dir, f"{experiment_name}/labels") # Labels are generated after the yolov5 run completeds 



# Function to convert xc, yc, w, h to x, y, w, h
def convert_bbox(xc, yc, w, h, img_width, img_height):
    # Convert relative coordinates to absolute
    abs_w = w * img_width
    abs_h = h * img_height
    abs_x = (xc * img_width) - (abs_w / 2)
    abs_y = (yc * img_height) - (abs_h / 2)  
    return abs_x, abs_y, abs_w, abs_h


# Function to run YOLOv5 detection on a single image
def run_detection(img_file):
    # command = [
    #     "python","detect.py",
    #     "--weights", weights_path,
    #     "--source", img_file,
    #     "--project", output_dir,
    #     "--name", experiment_name,  
    #     "--exist-ok",  
    #     "--save-txt",
    #     '--conf-thres', str(conf),
    #     "--save-conf"
    # ]
    command = [
        "python","detect.py",
        "--weights", weights_path,
        "--source", img_file,
        "--project", output_dir,
        "--name", experiment_name,  
        "--exist-ok",  
        "--save-txt",
        '--conf-thres', str(conf),
        "--save-conf"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {img_file}: {e}")

    # Clear CUDA cache to avoid memory fragmentation
    torch.cuda.empty_cache()

images_full_path = [os.path.join(input_dir,img) for img in IMAGES ] 

for image in images_full_path: 
    # File Structure for the RAW data that is used by the post processing 
    result = {
        'jsonData': {
            "result": [
                {
                    'boxes': [], 
                    'scores': [],  
                    'classes': []  
                }
            ],
            "metadata": "test"
        }
    } 
    img_name = image.split("/")[-1].split(".jpg")[0] 
    run_detection(image) 
    with Image.open(image) as img:
        # Get image size
        img_width, img_height = img.size 
    LABELS_PATH = os.path.join(LABELS_DIR, f"{img_name}.txt")
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as label_file:
            for line in label_file:
                data = line.strip().split()
                class_id = int(data[0])  # First value is the class
                xc, yc, w, h = map(float, data[1:5])  # Next four are the bounding box in xc, yc, w, h format
                score = data[-1]
                # Convert to x, y, w, h
                x, y, bbox_w, bbox_h = convert_bbox(xc, yc, w, h, img_width, img_height)
                
                # Append the values to the result dictionary
                result['jsonData']['result'][0]['boxes'].append([x, y, x+bbox_w, y+bbox_h])
                result['jsonData']['result'][0]['classes'].append(class_dict[class_id])
                result['jsonData']['result'][0]['scores'].append(float(score))  
            # Dump the RAW JSON 
            print(f"Creating the RAW JSONS for {img_name}")
            os.makedirs(os.path.join(output_dir, f"{experiment_name}/RAW"), exist_ok=True) 
            with open(f'{output_dir}/{experiment_name}/RAW/{img_name}.json','w') as fp: 
                json.dump(result, fp, indent=4) 
            post_process_json = postprocess_dit(result['jsonData']['result'][0])  
            
            os.makedirs(os.path.join(output_dir, f"{experiment_name}/post_process"), exist_ok=True)
            # Save the post_process json 
            print(f"Creating the POST PROCESSED JSONS for {img_name}")
            with open(f"{output_dir}/{experiment_name}/post_process/{img_name}.json",'w') as fp: 
                json.dump(post_process_json, fp, indent=4)
            
            
            if config['visualization']: 
                img = cv2.imread(image)
                post_process_img = visualize_prediction(post_process_json, img, raw=False) 
                os.makedirs(f"{output_dir}/{experiment_name}/post_process_img", exist_ok=True)
                cv2.imwrite(f"{output_dir}/{experiment_name}/post_process_img/{img_name}.jpg", post_process_img)
                print(f"Image: {img_name} processed successfully")
        