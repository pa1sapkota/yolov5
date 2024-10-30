from .post_process import postprocess_dit 
import json 
import os 
import cv2 


def read_json(file_path:str): 
    with open(file_path,'r') as fp: 
        json_data = json.load(fp)
    return json_data

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
        "Stamps":"green"
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

def visualize_prediction( json_data, img, raw=True): 
    if raw: 
        annotations = json_data['jsonData']['result'][0]
        bboxes = annotations['boxes']
        pred_cls = annotations['classes']
        pred_scores = annotations['scores'] 
        
        # plot the results for these 
        for i, class_name in  enumerate(pred_cls):
            x1, y1, x2, y2 = [int(x) for x in bboxes[i]]
        
                
            #  Color Mapping   
            color_name = label_color_map[class_name]
            color = color_name_to_rgb.get(color_name, (0, 0, 0))
            # Draw rectangle
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Adjusted coordinates
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_name + f"::::conf{round(pred_scores[i], 2)} ", (x1, y1 - 5), font, 0.5, color, 2, cv2.LINE_AA)
        return img 
    else : 
        # save for the postprocess output , make this uniform for both ! low priority task 
        annotations = json_data['dit']
        bboxes = annotations['boxes']
        pred_cls = annotations['classes']
        pred_scores = annotations['scores'] 
        
        # plot the results for these 
        for i, class_name in  enumerate(pred_cls):
            x1, y1, x2, y2 = [int(x) for x in bboxes[i]]
        
                
            #  Color Mapping   
            color_name = label_color_map[class_name]
            color = color_name_to_rgb.get(color_name, (0, 0, 0))
            # Draw rectangle
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Adjusted coordinates
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_name + f"::::conf{round(pred_scores[i], 2)}", (x1, y1 - 5), font, 0.5, color, 2, cv2.LINE_AA)
        return img 
