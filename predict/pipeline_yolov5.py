
import os
import subprocess
import torch
from concurrent.futures import ThreadPoolExecutor
import json 
# For the yolo the Indices start from the 0 
class_dict = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10:"Title"}


##### Make changes to these variable for running 
input_dir = "/home/ubuntu/yolov5/COCO/images/val2017"  # Directory containing images to run detection on
output_dir = "/home/ubuntu/yolov5/outputs"  # Directory to save the detection results
os.makedirs(output_dir, exist_ok=True)


weights_path = "/home/ubuntu/yolov5/runs/train/exp18/weights/best.pt"  # Path to the weights file
conf = "0.01"
experiment_name = "oct28_val_set"
IMG_TYPE = ('.jpg', '.png') # Endswith take tuple or string as arg




# Create the output directory if it doesn't exist
IMAGES = os.listdir(input_dir) 
IMAGES = sorted([img for img in IMAGES if img.endswith(IMG_TYPE)])  
LABELS_DIR = os.path.join(output_dir, f"{experiment_name}/labels") # Labels are generated after the yolov5 run completeds 


# read the json and then return the json parsed o/p
def read_json(file_path:str): 
    with open(file_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data

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
    input_path = os.path.join(input_dir, img_file)
    command = [
        "python","detect.py",
        "--weights", weights_path,
        "--source", input_path,
        "--project", output_dir,
        "--name", experiment_name,  # Name of the folder where results will be stored
        "--exist-ok",  # Allows results to overwrite existing files
        "--save-txt",
        '--conf-thres', conf,
        "--save-conf"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {img_file}: {e}")

    # Clear CUDA cache to avoid memory fragmentation
    torch.cuda.empty_cache()

# Get list of images
images = [img for img in sorted(os.listdir(input_dir)) if img.lower().endswith((".jpg", ".jpeg", ".png"))]

# Limit the number of workers to avoid memory overflow
max_workers = min(4, len(images))  # Set to 2 or adjust based on testing and available memory

# Use ThreadPoolExecutor to run detection in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(run_detection, images)

print("Detection completed and results saved to:", output_dir,"/",experiment_name) 




