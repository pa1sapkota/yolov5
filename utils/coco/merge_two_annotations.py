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
    max_image_id = 0
    max_annotation_id = 0


    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                combined_data['categories'] = data['categories']

                # Update images and annotations with unique IDs
                for image in data["images"]:
                    new_image_id = max_image_id + 1
                    image["id"] = new_image_id
                    combined_data["images"].append(image)
                    max_image_id = new_image_id

                for annotation in data["annotations"]:
                    new_annotation_id = max_annotation_id + 1
                    annotation["id"] = new_annotation_id
                    annotation["image_id"] += max_image_id  # Update image_id to correspond to new image IDs
                    combined_data["annotations"].append(annotation)
                    max_annotation_id = new_annotation_id

    # Write combined data to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined JSON written to {output_file}")

# Usage
json_directory = 'utils/coco/annotation/json'  # Directory containing JSON files
output_filename = 'utils/coco/annotation/annot.json'  # Output file name
combine_coco_jsons(json_directory, output_filename)
