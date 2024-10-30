import json 
import os 
from PIL import Image
import cv2 


IMG_DIR = "/home/ubuntu/yolov5/outputs/oct25_finance"
IMG_TYPE = ('.jpg', '.png') # Endss with take tuple or string as arg
LABELS_DIR = os.path.join(IMG_DIR, "labels")
output_name = "prediction_to_pipeline_format"
os.makedirs(f"/home/ubuntu/yolov5/outputs/oct25_finance/{output_name}",exist_ok=True)

class_dict = {0: "Caption",
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


IMAGES = os.listdir(IMG_DIR) 
IMAGES = sorted([img for img in IMAGES if img.endswith(IMG_TYPE)])  


# Function to convert xc, yc, w, h to x, y, w, h
def convert_bbox(xc, yc, w, h, img_width, img_height):
    # Convert relative coordinates to absolute
    abs_w = w * img_width
    abs_h = h * img_height
    abs_x = (xc * img_width) - (abs_w / 2)
    abs_y = (yc * img_height) - (abs_h / 2)  
    return abs_x, abs_y, abs_w, abs_h

# Loop through the images
for idx, image in enumerate(IMAGES):
    result = {
        'jsonData': {
            "result": [
                {
                    'boxes': [],  # To store converted bounding boxes
                    'scores': [],  # Placeholder for scores
                    'classes': []  # Placeholder for classes
                }
            ],
            "config": "test"
        }
    }
    
    img_name = image
    full_image_path = os.path.join(IMG_DIR, image)
    
    with Image.open(full_image_path) as img:
        # Get image size
        img_width, img_height = img.size
    
    # Read the corresponding label file (assumes label file has the same name as the image with .txt extension)
    label_file_path = os.path.join(LABELS_DIR, os.path.splitext(image)[0] + '.txt')
    print(f"Processing:{label_file_path}")
    if os.path.exists(label_file_path):
        print(label_file_path)
        with open(label_file_path, 'r') as label_file:
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
                result['jsonData']['result'][0]['scores'].append(float(score))  # Placeholder for scores
    
    with open(f'/home/ubuntu/yolov5/outputs/oct22_finance/{output_name}/{img_name}.json','w') as fp: 
        json.dump(result, fp, indent=4)
        