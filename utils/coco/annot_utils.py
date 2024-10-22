import json 
import os 
import argparse 

doclaynet_categories = [
    {
      "supercategory": "Caption",
      "id": 1,
      "name": "Caption"
    },
    {
      "supercategory": "Footnote",
      "id": 2,
      "name": "Footnote"
    },
    {
      "supercategory": "Formula",
      "id": 3,
      "name": "Formula"
    },
    {
      "supercategory": "List-item",
      "id": 4,
      "name": "List-item"
    },
    {
      "supercategory": "Page-footer",
      "id": 5,
      "name": "Page-footer"
    },
    {
      "supercategory": "Page-header",
      "id": 6,
      "name": "Page-header"
    },
    {
      "supercategory": "Picture",
      "id": 7,
      "name": "Picture"
    },
    {
      "supercategory": "Section-header",
      "id": 8,
      "name": "Section-header"
    },
    {
      "supercategory": "Table",
      "id": 9,
      "name": "Table"
    },
    {
      "supercategory": "Text",
      "id": 10,
      "name": "Text"
    },
    {
      "supercategory": "Title",
      "id": 11,
      "name": "Title"
    }
  ] 

def get_labels_id_map(): 
    """
    Get the Categories and their Corresponding ids for the Default Doclaynet Categories
    """
    result_map = dict() 
    for categories in doclaynet_categories: 
        result_map[categories["name"]] = categories['id']
    return result_map
        

def read_json(json_path:str) : 
    with open(json_path) as fp: 
        json_data = json.load(fp)
    return json_data 



def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a JSON file and specify output directory.")
    parser.add_argument('--json', type=str, required=True, help="Path to the JSON file")
    parser.add_argument('--o', '--output_dir', type=str, required=True, help="Output directory")

    args = parser.parse_args()
    return args

    
def update_json(json_file, output_dir, classes_to_ignore= ["Stamps", "Handwriting", "Signature"]): 
    final_json_data = dict() 
    final_json_data['images'] = json_file['images']
    final_json_data['categories'] = doclaynet_categories # we will be using the default doclaynet settings for maintaining uniformity 
    final_json_data['annotations'] = [] 
    # need to change all the annotations ids since they might not always has same category -id map as the doclaynet 
    default_doclaynet_categories_id_map = get_labels_id_map()
    annot_id_to_category_map = dict() 
    for annot_cateogry in json_file['categories']: 
        annot_id_to_category_map[annot_cateogry['id']] = annot_cateogry['name'] 
    # now changing all the Annotaions and updating to maintain the uniformity for the annotations 
    for annotation in json_file['annotations']: 
        if annot_id_to_category_map[annotation['category_id']] not in classes_to_ignore: 
            category_labels = annot_id_to_category_map[annotation['category_id']] # get the category id 
            new_id = default_doclaynet_categories_id_map[category_labels]         
            annotation['category_id'] = new_id
            final_json_data['annotations'].append(annotation) 
    
    # Before Sending the final annoations we will combine the Section-headers 
    
        
    return final_json_data

def main(): 
    args = parse_arguments()

    # Extract the variables
    json_file_path = args.json
    output_dir = args.o
    
    json_file = read_json(json_file_path)
    final_json = update_json(json_file, output_dir)
    with open(f"{output_dir}/final.json", 'w') as fp: 
        json.dump(final_json, fp, indent=4)
    

if __name__=="__main__": 
    
    print("Started Conversion")
    main() 