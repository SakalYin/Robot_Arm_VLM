import numpy as np
from model.qwenvl.utils import extract_position_and_orientation
import torch

def euclidean_distance(predicted, truth, error_value=2.0):
    distances = []
    for p1, p2 in zip(predicted, truth):
        # Skip if p1 or p2 is invalid
        if p1 is None or p2 is None or len(p1) != 3 or len(p2) != 3:
            distances.append(error_value)
            continue
        
        d = ((float(p1[0]) - float(p2[0])) ** 2 + 
             (float(p1[1]) - float(p2[1])) ** 2 + 
             (float(p1[2]) - float(p2[2])) ** 2) ** 0.5
        distances.append(d)
    return np.mean(distances)

def mae_xyz(predicted, truth, error_value=[0.95, 1.85, 0.5]):
    errors_x = []
    errors_y = []
    errors_z = []

    for p1, p2 in zip(predicted, truth):
        # Skip if p1 or p2 is invalid
        if p1 is None or p2 is None or len(p1) != 3 or len(p2) != 3:
            errors_x.append(error_value[0])
            errors_y.append(error_value[1])
            errors_z.append(error_value[2])
            continue
        
        errors_x.append(abs(float(p1[0]) - float(p2[0])))
        errors_y.append(abs(float(p1[1]) - float(p2[1])))
        errors_z.append(abs(float(p1[2]) - float(p2[2])))

    mae_x = np.mean(errors_x)
    mae_y = np.mean(errors_y)
    mae_z = np.mean(errors_z)

    return mae_x, mae_y, mae_z

class RobotArmMetric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, eval_preds):
        tokenizer = self.tokenizer
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        print(dir(eval_preds))
        print(eval_preds.label_ids)
        print(eval_preds.predictions)
        print(eval_preds.inputs)
        print(eval_preds.label_ids.shape)
        print(eval_preds.predictions.shape)
        print(eval_preds.inputs.shapes)

        # Replace -100 in the preds as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, padding_side='left')

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, padding_side='left')

        # print(decoded_preds)

        # extract values (fix to use batch)
        predicted_pos = [pos for text in decoded_preds for (pos,ori,obj) in [extract_position_and_orientation(text)]]
        label_pos = [pos for text in decoded_labels for (pos,ori,obj) in [extract_position_and_orientation(text)]]
        error_indices = [i for i, pos in enumerate(predicted_pos) if (pos is None) or (len(pos) < 3)]
        corrected_pos_ratio = len(error_indices) / len(predicted_pos) if len(predicted_pos) > 0 else 0.0
        print(predicted_pos)
        print(label_pos)
        predicted_ori = [ori for text in decoded_preds for (pos,ori,obj) in [extract_position_and_orientation(text)]]
        label_ori = [ori for text in decoded_labels for (pos,ori,obj) in [extract_position_and_orientation(text)]]

        # euclidean
        distances = euclidean_distance(predicted_pos, label_pos, error_indices)
        # print(predicted_pos, error_indices, distances)
        # mae
        mae_x, mae_y, mae_z = mae_xyz(predicted_pos, label_pos, error_value=[0.95, 1.85, 0.5])

        return {
            'euclidean': round(distances,4),
            'mae_x': round(mae_x, 4),
            'mae_y': round(mae_y, 4),
            'mae_z': round(mae_z,4),
            'corrected_pos_ratio': round(corrected_pos_ratio, 4)
            }