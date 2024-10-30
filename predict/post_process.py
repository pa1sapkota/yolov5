import datetime

import os
from PIL import Image
import io
import base64
import json
import cv2
import logging

# Configure logging
logging.basicConfig(
    filename="debug_DSR.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
debug = False  # Set the Debug Flag


def get_area(res):
    bbox = res[0]
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)

    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    area_intersection = intersection_w * intersection_h

    iou = area_intersection / (area_box1 + area_box2 - area_intersection)
    return iou, (area_intersection / area_box1, area_intersection / area_box2)


def remove_smaller_bbox(res1, res2):
    area1, area2 = get_area(res1), get_area(res2)
    if area1 > area2:
        return res2
    else:
        return res1


def postprocess_(
    res, upper_intersection_threshold=0.9, lower_intersection_threshold=0.3
):
    res_to_remove = []
    ignored_classes = ["Picture", "Stamps", "Signature"]
    res_to_remove = [
        res[i]
        for i in range(len(res))
        if res[i][2] in ignored_classes
        or is_zero_dimension(res[i][0])
        or is_abnormal_ratio(res[i][0])
    ]  # Checking if the results are in the Ignored classes and removing abnormal cases as well

    confidence_thres = 0.7
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if res[i] in res_to_remove or res[j] in res_to_remove:
                continue

            if (res[i][2] in ignored_classes) or (res[j][2] in ignored_classes):
                continue
            iou, intersection_area = calculate_iou(res[i][0], res[j][0])

            if (
                intersection_area[0] > upper_intersection_threshold
                or intersection_area[1] > upper_intersection_threshold
            ):
                if res[i][2] == "Table" and res[j][2] == "Table":
                    if res[i][1] > confidence_thres and res[j][1] > confidence_thres:
                        if get_area(res[i]) > get_area(res[j]):
                            res_to_remove.append(res[j])
                            continue
                        else:
                            res_to_remove.append(res[i])
                            break
                    else:
                        if res[i][1] > res[j][1]:
                            res_to_remove.append(res[j])
                            if debug:
                                logging.debug(
                                    f"Removed {res[j]} because it has lower confidence among Tables"
                                )
                            continue
                        else:
                            res_to_remove.append(res[i])
                            if debug:
                                logging.debug(
                                    f"Removed {res[i]} because it has lower confidence among Tables"
                                )
                            break

                elif (res[i][2] == "Table") ^ (res[j][2] == "Table"):
                    if res[i][2] == "Table":
                        
                        status = handle_table_and_nontable(
                            table_idx=i, res=res, res_to_remove=res_to_remove
                        )
                        if status == "break":
                            break
                        elif status == "continue":
                            continue
                    if res[j][2] == "Table":
                        status = handle_table_and_nontable(
                            table_idx=j, res=res, res_to_remove=res_to_remove
                        )
                        if status == "break":
                            break
                        elif status == "continue":
                            continue

                elif (res[i][2] != "Table") and (res[j][2] != "Table"): 
                    _, (intersection_area_res_i, intersection_area_res_j) = (
                        calculate_iou(res[i][0], res[j][0])
                    )
                    # If the Text Predictions are entirely inside the another Predictions
                    # Since we had very low confidence predictions so to choose
                    if (
                        (
                            intersection_area_res_i > 0.99
                            or intersection_area_res_j > 0.99
                        )
                        and (res[i][1] > 0.7)
                        and (res[j][1] > 0.7)
                    ):
                        removed_res = remove_smaller_bbox(res[i], res[j])
                        res_to_remove.append(removed_res)
                        continue
                    else:
                        # Remove the Prediction based on Scores
                        if res[i][1] >= res[j][1]:
                            res_to_remove.append(res[j])


                            continue
                        else:
                            res_to_remove.append(res[i])
                            break

            elif (
                intersection_area[0] > lower_intersection_threshold
                or intersection_area[1] > lower_intersection_threshold
            ):
                if res[i][2] != "Table" and res[j][2] != "Table":
                    # Here we have to make logic such that it will take the bigger area within certain conf after that
                    # it will filter based on the Score
                    # For GlobalIME : 0.4 $ confident we choose the bigger area
                    # Experimenting , for choosing the bigger area for the prediction lets choose bigger area given they are both 80% confident 
                    if res[i][1] > 0.8 and res[j][1] > 0.8:
                        removed_res = remove_smaller_bbox(res[i], res[j])
                        res_to_remove.append(removed_res)

                        continue
                    else:
                        # Remove based on the Score
                        if res[i][1] > res[j][1]:
                            res_to_remove.append(res[j])
                            continue
                        else:
                            res_to_remove.append(res[i])
                            continue

    for remove_res in res_to_remove:
        if remove_res in res:
            res.remove(remove_res)
    return res


def has_significant_overlap(pred, final_result):
    # This functions check if the prediction has overlap in any of the final_result from the previous post_procession
    # Generally retains non-Table Prediction for now-> Getting Note Table if missed
    bbox, _, pred_class = pred
    result_overlap = [
        calculate_iou(bbox, result[0])[1]
        for result in final_result
        if get_area(pred) < get_area(result)
        and calculate_iou(bbox, result[0])[1][0] > 0.1
        and calculate_iou(bbox, result[0])[1][0]
        < 0.5  # Ensures the portion of bbox is  less then 50% inside results
        and (pred_class != "Table" and result[2] == "Table")
    ]
    if result_overlap:
        return all(x[0] < 0.5 for x in result_overlap)


def has_minimal_overlap(pred, final_result, thres):
    return all(
        x[0] < thres and x[1] < thres
        for x in [calculate_iou(pred[0], result[0])[1] for result in final_result]
    )


def filter_overlapping_predictions(missed_predictions):
    predictions_to_remove = []

    for i, pred_i in enumerate(missed_predictions):
        for j in range(i + 1, len(missed_predictions)):
            pred_j = missed_predictions[j]
            if pred_i in predictions_to_remove or pred_j in predictions_to_remove:
                continue
            _, (area1, area2) = calculate_iou(pred_i[0], pred_j[0])
            if area1 > 0.9 or area2 > 0.9:
                if pred_i[1] > pred_j[1]:
                    predictions_to_remove.append(pred_j)
                else:
                    predictions_to_remove.append(pred_i)

    return [pred for pred in missed_predictions if pred not in predictions_to_remove]


def is_abnormal_ratio(bbox):
    """Checks the Height and width Ratio and returns True for Abnormal Ratio

    Args:
        bbox (list): bbox of the prediction
        - x1,y1,x2,y2
    """
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    if width / height >= 100:
        return True
    return False


def is_zero_dimension(bbox):
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    return width <= 0 or height <= 0


def remove_based_on_confidence(res1, res2):
    return res2 if res1[1] > res2[1] else res1


def remove_based_on_area(res1, res2):
    return res2 if get_area(res1) > get_area(res2) else res1


def handle_table_and_nontable(table_idx, res, res_to_remove):
    i = table_idx
    # will check this logic --->
    # preds_inside_table = [
    #     x for x in res if calculate_iou(res[i][0], x[0])[1][1] > 0.7 and x != res[i]
    # ]
    preds_inside_table = [
        x for x in res if calculate_iou(res[i][0], x[0])[1][1] > 0.7 and x != res[i]
    ]
    if len(preds_inside_table) > 1:
        inside_table_preds_dict = {}
        for pred in preds_inside_table:
            if pred[2] not in inside_table_preds_dict.keys():
                inside_table_preds_dict[pred[2]] = [pred]
            else:
                inside_table_preds_dict[pred[2]].append(pred)
        # Class which has the max ratio of predictions  among predictions  inside the Table
        pred_with_high_ratio = max(
            inside_table_preds_dict.keys(),
            key=lambda k: len(inside_table_preds_dict[k]),
        )

        # Get the Length of the Preditions have more ratio inside the Table
        if len(inside_table_preds_dict[pred_with_high_ratio]) > 4:
            confidences = [
                value
                for key, value in inside_table_preds_dict.items()
                if key == pred_with_high_ratio
            ]
            # Extracting only the confidence scores
            scores = [
                confidence for sublist in confidences for (_, confidence, _) in sublist
            ]
            avg_confidence = sum(scores) / len(scores)
            if (avg_confidence > res[i][1]) and res[i][1] < 0.5:
                res_to_remove.append(res[i])
                if debug:
                    logging.debug(
                        f"Table {res[i]} removed because avg confidence of inside preds is higher"
                    )
                # break # Handle later
                return "break"
            else:
                # remove all the predicions that is inside the Table
                res_to_remove.extend(preds_inside_table)
                if debug:
                    logging.debug(
                        f"Preds inside Table{preds_inside_table} removed because they have lower avg confidence"
                    )
                # continue # Handle Later
                return "continue"
        else:
            # remove these pred inside Table if those are less than 4
            # only remove if the confidence of the table is more than 20%
            if res[i][1] > 0.2:
                res_to_remove.extend(preds_inside_table)
                if debug:
                    logging.debug(
                        f"Preds inside Table: {preds_inside_table} removed because they are less than 4"
                    )
                # continue #Handle later
                return "continue"
            else:
                res_to_remove.append(res[i])
                if debug:
                    logging.debug(
                        f"Table{res[i]} removed as this was not confident Enough"
                    )
    elif len(preds_inside_table) == 1:
        _, (intersection_area_by_table, intersection_area_by_pred) = calculate_iou(
            res[i][0], preds_inside_table[0][0]
        )
        # This checks if both of them overlap intersecting area of both is greater than 70%
        if intersection_area_by_table > 0.7 and intersection_area_by_pred > 0.7:
            # Check the Prediction score
            if res[i][1] > preds_inside_table[0][1]:
                res_to_remove.extend(preds_inside_table)
                if debug:
                    logging.debug(
                        f"Single pred inside Table: {preds_inside_table} removed because Table's confidence is higher"
                    )
                # continue
                return "continue"

            else:
                res_to_remove.append(res[i])
                if debug:
                    logging.debug(
                        f"Table: {res[i]} removed because single pred inside has higher confidence"
                    )
                # break
                return "break"
        else:
            # remove the predictions inside
            res_to_remove.extend(preds_inside_table)


def postprocess_second_phase(res, final_result, class_to_ignore):
    """
    Run the post processing for the 2nd time to retrieve missing low confidence predictions
    or predictions that were removed during the first postprocessing.

    Args:
        res (list): List of tuples containing raw predictions with:
            - bounding box (list)
            - confidence of prediction (float)
            - class of prediction (str)

        final_result (list): List of tuples containing post-processed results with:
            - bounding box (list)
            - confidence of prediction (float)
            - class of prediction (str)

        class_to_ignore (list): List of strings representing the classes to ignore.

    Returns:
        list: List of missed predictions after the second phase of post-processing.
    """

    missed_predictions = []
    thres = 0.3

    for pred in res:
        bbox, _, pred_class = pred
        if is_zero_dimension(bbox) or is_abnormal_ratio(bbox):
            continue
        elif pred_class in class_to_ignore:
            continue
        if has_significant_overlap(pred, final_result):
            missed_predictions.append(pred)
        elif has_minimal_overlap(pred, final_result, thres):
            missed_predictions.append(pred)

    return filter_overlapping_predictions(missed_predictions)


def postprocess_dit(unprocessed_dict):
    pred_to_ignore = ["Picture", "Stamps", "Signature"]
    threshold = 0.15
    upper_intersection_threshold = 0.9
    lower_intersection_threshold = 0.3

    res_dict = {}
    res = []
    all_res = []
    # import pdb ; pdb.set_trace()
    # unprocessed_dict = unprocessed_dict['dit']['results'][0]

    # Getting the results from the unprocessed dit
    boxes = unprocessed_dict["boxes"]
    classes = unprocessed_dict["classes"]
    scores = unprocessed_dict["scores"]

    for idx in range(len(boxes)):
        all_res.append((boxes[idx], scores[idx], classes[idx]))
        if scores[idx] > threshold:
            res.append((boxes[idx], scores[idx], classes[idx]))
    final_result = postprocess_(
        res, upper_intersection_threshold, lower_intersection_threshold
    )
    missed_predictions = postprocess_second_phase(all_res, final_result, pred_to_ignore)

    if missed_predictions:
        final_result.extend(missed_predictions)

    # final_result = res
    boxes, classes, scores = [], [], []
    for idx, tab in enumerate(final_result):
        boxes.append(tab[0])
        classes.append(tab[2])
        scores.append(tab[1])

    # res_dict['jsonData'] = {
    #     "boxes": boxes,
    #     "classes": classes,
    #     "scores": scores,
    #     "image" : unprocessed_dict['image']
    #     # 'LEN': unprocessed_dict['LEN'],
    #     # 'IMAGE_SHAPE': unprocessed_dict['IMAGE_SHAPE']
    #  }
    # return res_dict
    sorted_data = sorted(zip(boxes, scores, classes), key=lambda x: x[0][1])
    try:
        sorted_boxes, sorted_scores, sorted_classes = zip(*sorted_data)
    except ValueError:
        sorted_boxes = []
        sorted_scores = []
        sorted_classes = []

    res_dict["dit"] = {
        "boxes": sorted_boxes,
        "classes": sorted_classes,
        "scores": sorted_scores,
    }
    return res_dict


def visualize_dit_post_processed(jsonData):
    from PIL import Image
    import numpy as np
    import base64
    import cv2
    from io import BytesIO

    # jsonData = jsonData['dit']
    try:
        jsonData = jsonData["dit"]["results"][0]
    except KeyError:
        import pdb

        pdb.set_trace()
    image = jsonData["image"]

    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, class_name in enumerate(jsonData["classes"]):
        label_color_map = {
            "Caption": "gray",
            "Footnote": "pink",
            "Formula": "yellow",
            "List-item": "orange",
            "Pagefooter": "purple",
            "Page-header": "cyan",
            "Picture": "blue",
            "Section-header": "blue",
            "Table": "brown",
            "Text": "green",
            "Title": "red",
            "Handwriting": "red",
            "Stamps": "green",
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
            "black": (0, 0, 0),
        }
        visualize_all_labels = True
        if (
            class_name == "Picture"
            or class_name == "Handwriting"
            or class_name == "Stamps"
            or visualize_all_labels
        ):
            x1, y1, x2, y2 = [int(x) for x in jsonData["boxes"][i]]

            #  Color Mapping
            color_name = label_color_map[class_name]
            color = color_name_to_rgb.get(color_name, (0, 0, 0))
            #     color = label_color_map[class_name]

            # Draw rectangle
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Adjusted coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                image,
                class_name + f"::::conf{round(jsonData['scores'][i], 2)}",
                (x1, y1 - 5),
                font,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return image


def visualize_dit(jsonData):
    from PIL import Image
    import numpy as np
    import base64
    import cv2
    from io import BytesIO

    try:
        jsonData = jsonData["dit"]["results"][0]
    except KeyError:
        import pdb

        pdb.set_trace()
    image = jsonData["image"]

    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, class_name in enumerate(jsonData["classes"]):
        label_color_map = {
            "Caption": "gray",
            "Footnote": "pink",
            "Formula": "yellow",
            "List-item": "orange",
            "Pagefooter": "purple",
            "Page-header": "cyan",
            "Picture": "blue",
            "Section-header": "blue",
            "Table": "brown",
            "Text": "green",
            "Title": "red",
            "Handwriting": "red",
            "Stamps": "green",
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
            "black": (0, 0, 0),
        }
        visualize_all_labels = True
        if (
            class_name == "Picture"
            or class_name == "Handwriting"
            or class_name == "Stamps"
            or visualize_all_labels
        ):
            x1, y1, x2, y2 = [int(x) for x in jsonData["boxes"][i]]

            #  Color Mapping
            color_name = label_color_map[class_name]
            color = color_name_to_rgb.get(color_name, (0, 0, 0))
            #     color = label_color_map[class_name]

            # Draw rectangle
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Adjusted coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                image,
                class_name + f"::::conf{round(jsonData['scores'][i], 2)}",
                (x1, y1 - 5),
                font,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return image