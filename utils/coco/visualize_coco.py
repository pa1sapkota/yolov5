import json 
import os 
import shutil 
import cv2  
from collections import defaultdict



label_color_map = {
        "Caption": "gray",
        "Footnote": "pink",
        "Formula": "yellow",
        "List-item": "orange",
        "Page-footer": "purple",
        "Page-header": "cyan",
        "Picture": "blue",
        "Section-header": "blue",
        "Table": "brown",
        "Text": "green",
        "Title": "red",
        "Handwriting":"red",
        "Stamps":"green",
        "Signature" : 'red'
}
color_name_to_rgb = {
        "gray": (128, 128, 128),
        "pink": (255, 182, 193),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "blue": (0, 0, 255),
        "brown": (165, 42, 42),
        "green": (0, 128, 0),
        "red": (255, 0, 0),
        "black": (0, 0, 0)
}



def read_json(json_path:str): 
    with open(json_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data 

def draw_bbox(img, annotations, categories): 
    # Draw bounding boxes and category names
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        category_name = categories[category_id]
        color = color_name_to_rgb[label_color_map[category_name]]

        # Calculate top-left and bottom-right coordinates
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height

        # Draw bounding box (red, thickness = 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Put category name above the bounding box (white text, red background)
        cv2.putText(img, category_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2, lineType=cv2.LINE_AA)
    return img  


if __name__ == "__main__": 
    IMG_DIR = ""
    JSON_PATH = ""
    save_path = ""
    os.makedirs(save_path, exist_ok=True)
    json_file = read_json(JSON_PATH) 
    categories = dict() 
    for cat in json_file['categories']: 
        categories[cat['id']] = cat['name']
    imageIdToName = dict() 
    for image in json_file['images']: 
        imageIdToName[image['id']] = image['file_name']
    # Get the image_id and annotations map 
    imageidtoannotationmap = defaultdict(list)
    for annotation in json_file['annotations']: 
        imageidtoannotationmap[annotation['image_id']].append(annotation) 
        
    for img_id, annotation in imageidtoannotationmap.items(): 
        image_name = imageIdToName[img_id]
        full_image_path = os.path.join(IMG_DIR, image_name)
        img = cv2.imread(full_image_path)
        annot_image = draw_bbox(img, annotation, categories)
        cv2.imwrite(f"{save_path}/{image_name}", img)