from utils import update_annotation , merge_section_headers
import json 
import os 
from collections import defaultdict
from PIL import Image


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def read_json(json_path:str): 
    with open(json_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data 





if __name__ == "__main__": 
    annotFilePath = ""
    image_dir = ""
    document_name = ""
    sample_type = "train"
    output_json_dir = f"output/{document_name}"
    os.makedirs(output_json_dir, exist_ok=True)
    
    annotationFile = read_json(annotFilePath)
    annotationByImage = defaultdict(list)
    post_processed_json = dict() 
    post_processed_json['categories'] = annotationFile['categories']
    post_processed_json['images'] = annotationFile['images']
    post_processed_json['annotations'] = []
    imageNamebyId = defaultdict(str)
    for image in annotationFile['images']: 
        # Create the image id and the annotation dict 
        imageNamebyId[image['id']] = (image['file_name'])
    for annotation in annotationFile['annotations']: 
        # Create the image id and the annotation dict 
        annotationByImage[annotation['image_id']].append(annotation) 
    # Handle the Each images and their annotation 
    for image_id, annotations in annotationByImage.items():
        print(f"Processing Image ID: {image_id}")
        width, height = get_image_size(os.path.join(image_dir, imageNamebyId[image_id]))
        # Update each annotation based on the label change dictionary
        updated_annotations = [update_annotation(ann) for ann in annotations] 
        combined_annotations = merge_section_headers(updated_annotations,width, height)   
        post_processed_json['annotations'].extend(combined_annotations)
    
    with open(f'{output_json_dir}/{sample_type}.json','w') as fp: 
        json.dump(post_processed_json, fp, indent=4)

        

