
labels_change_dict = {
                "Page-header": "Section-header", 
                "Title": "Section-header", 
                "Footnote": "Text", 
                "Page-footer": "Text", 
}

category_dict = {
        1: 'Caption',
        2: 'Footnote',
        3: 'Formula',
        4: 'List-item',
        5: 'Page-footer',
        6: 'Page-header',
        7: 'Picture',
        8: 'Section-header',
        9: 'Table',
        10: 'Text',
        11: 'Title'}



def get_segmentation(bbox): 
    """Getting the Segmentation coordinate for the rectangular bbox 

    Args:
        bbox (list): bbox coordinates

    Returns:
        segmentation_points(list): rectangular segmentation points
    """
    segmentation_points = [] 
    x, y, width, height = bbox
    x_min,y_min = x,y
    x_max = x + width
    y_max = y + height
    segmentation_points.append([x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min])
    return segmentation_points

def update_annotation(annotation): 
    # Update the annotation by modifying the category_id based on the label change dict
    category_id = annotation['category_id']
    current_label = category_dict.get(category_id)

    # Check if the current label needs to be changed
    ## --> Need to work on how the Page-headers 
    x,y,w,h = annotation['bbox']
    if annotation['category_id'] == 6 and h/w > 5 : # need to change the height / weight ratio for the page-header examples 
        # Don't Change the annotation 
        return annotation  
        
    if current_label in labels_change_dict:
        new_label = labels_change_dict[current_label]
        # Find the new category_id from the updated label
        new_category_id = [k for k, v in category_dict.items() if v == new_label]
        if new_category_id:
            annotation['category_id'] = new_category_id[0]
    
    return annotation


def bigger_bbox(bbox1, bbox2): 
    if bbox1[2] > bbox2[2]: 
        return bbox1, bbox2
    else : 
        return bbox2, bbox1
def merge_section_headers(annotations, image_width, image_height, category_id=8, y_threshold_ratio=0.014, x_threshold_ratio = 0.025):
    # Check if two boxes are close enough to merge based on relative thresholds
    def is_close_enough(bbox1, bbox2, image_width, image_height,y_threshold_ratio,x_threshold_ratio):
        # Calculate dynamic thresholds based on image size
        y_threshold = y_threshold_ratio * image_height
        x_threshold = x_threshold_ratio * image_width
                # print(y_threshold)
        # print(x_threshold)
        # Check proximity in both x and y directions
        y_close = abs(bbox1[1] + bbox1[3] - bbox2[1]) <= y_threshold
        
        
        # Get the bigger bounding box 
        b_bbox, s_bbox = bigger_bbox(bbox1, bbox2)

        # Calculate the adjusted ranges for the larger bounding box
        b_bbox_start = b_bbox[0] - x_threshold  # Adjusted start with threshold
        b_bbox_end = b_bbox[0] + b_bbox[2] + x_threshold  # Adjusted end with threshold

        # Calculate the range for the smaller bounding box
        s_bbox_start = s_bbox[0]
        s_bbox_end = s_bbox[0] + s_bbox[2]
        between_bigger_bbox = False 
        if (b_bbox_start <= s_bbox_start) and (b_bbox_end >= s_bbox_end): 
            between_bigger_bbox= True 
        
        return y_close and between_bigger_bbox

    # Merging bounding boxes
    def merge_bboxes(bbox1, bbox2):
        x_min = min(bbox1[0], bbox2[0])
        y_min = min(bbox1[1], bbox2[1])
        x_max = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_max = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        return [x_min, y_min, x_max - x_min, y_max - y_min]



    # Sort annotations by y1 (bbox[1])
    section_headers = sorted(
        [ann for ann in annotations if ann['category_id'] == category_id],
        key=lambda x: x['bbox'][1]
    )

    merged_annotations = []
    i = 0
    while i < len(section_headers):
        current_ann = section_headers[i]
        current_bbox = current_ann['bbox']
        current_seg = current_ann['segmentation'][0]
        merged = False

        # Check if next annotation is close enough in both x and y directions
        while i + 1 < len(section_headers):
            next_ann = section_headers[i + 1]
            next_bbox = next_ann['bbox']
            if is_close_enough(current_bbox, next_bbox,image_width, image_height, y_threshold_ratio, x_threshold_ratio):
                # Merge bounding boxes and segmentations
                current_bbox = merge_bboxes(current_bbox, next_bbox) 
                
                current_seg = get_segmentation(current_bbox)
                i += 1
                merged = True
            else:
                break

        # Add the merged annotation back
        merged_annotations.append({
            'id': current_ann['id'],
            'image_id': current_ann['image_id'],
            'category_id': category_id,
            'bbox': current_bbox,
            'segmentation': current_seg,
            'area': current_bbox[2] * current_bbox[3],
        })
        i += 1
        # merged_annotations.append({
        #     'id': current_ann['id'],
        #     'image_id': current_ann['image_id'],
        #     'category_id': category_id,
        #     'bbox': current_bbox,
        #     'segmentation': current_seg,
        #     'area': current_bbox[2] * current_bbox[3],
        #     'iscrowd': current_ann['iscrowd'],
        #     'precedence': current_ann['precedence']
        # })
        # i += 1

    # Return the non-section-header annotations unchanged, along with the merged annotations
    return [ann for ann in annotations if ann['category_id'] != category_id] + merged_annotations