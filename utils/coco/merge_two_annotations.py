# Scripts to merge two coco annoations

import os
import json

def combine_coco_jsons(json_dir, output_file):
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Initialize max IDs
    total_max_image_id = 0
    total_max_annotation_id = 0

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                combined_data['categories'] = data['categories']

                # Update images and annotations with unique IDs
                for image in data["images"]:
                    new_image_id = image["id"] + 1 + total_max_image_id
                    image["id"] = new_image_id
                    combined_data["images"].append(image)

                for annotation in data["annotations"]:
                    new_annotation_id = annotation["id"] + 1 + total_max_annotation_id
                    annotation["id"] = new_annotation_id
                    annotation["image_id"] = annotation["image_id"] + 1 + total_max_image_id  # Update image_id to correspond to new image IDs
                    combined_data["annotations"].append(annotation)
        total_max_image_id += len(data["images"]) 
        total_max_annotation_id += len(data["annotations"])

    # Write combined data to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined JSON written to {output_file}")

# Usage
json_directory = 'downloads/datasets/json_files'  # Directory containing JSON files
output_filename = 'downloads/datasets/json_files/final.json'  # Output file name
combine_coco_jsons(json_directory, output_filename)
