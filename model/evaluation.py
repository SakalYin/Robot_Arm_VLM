import numpy as np
from sklearn.metrics import classification_report
from scipy.spatial.transform import Rotation as R
import sys
from model.qwenvl.utils import batch_inference, extract_position_and_orientation
import pandas as pd
import re

def euclidean_distance(predicted, truth):
    distances = [abs(((float(p1[0]) - float(p2[0]))**2 + (float(p1[1]) - float(p2[1]))**2 + (float(p1[2]) - float(p2[2]))**2) ** 0.5)
                    for p1,p2 in zip(predicted, truth)]
    return np.mean(distances)

def mae(predicted, truth):
    ae = [abs(float(p1) - float(p2)) for p1,p2 in zip(predicted, truth)]
    mae = np.mean(ae)
    return mae

def mae_xyz(predicted, truth):
    try:
        predicted = predicted.values.tolist()
    except:
        pass
    try:
        truth = truth.values.tolist()
    except:
        pass
    
    predicted, truth = np.array(predicted).T, np.array(truth).T
    mae_x = mae(predicted[0], truth[0])
    mae_y = mae(predicted[1], truth[1])
    mae_z = mae(predicted[2], truth[2])
    return mae_x, mae_y, mae_z

def r2_score(predicted, truth):
    rss = sum([(float(p1) - float(p2))**2 for p1,p2 in zip(predicted, truth)])
    mean = np.mean([float(p) for p in truth])
    tss = sum([(float(p1) - mean)**2 for p1 in truth])
    if tss == 0:
        return 0
    else:
        r2_score = 1 - (rss/tss)
        return r2_score

def r2_score_xyz(predicted, truth):
    try:
        predicted = predicted.values.tolist()
    except:
        pass
    try:
        truth = truth.values.tolist()
    except:
        pass
    
    predicted, truth = np.array(predicted).T, np.array(truth).T
    r2_score_x = r2_score(predicted[0], truth[0])
    r2_score_y = r2_score(predicted[1], truth[1])
    r2_score_z = r2_score(predicted[2], truth[2])
    return r2_score_x, r2_score_y, r2_score_z

def mean_quaternion_error(predicted, truth):
    try:
        predicted = predicted.values.tolist()
    except:
        pass
    try:
        truth = truth.values.tolist()
    except:
        pass
    
    rotation_error_deg = []
    for i in range(len(predicted)):
        r_truth = R.from_quat(truth[i])   # (x, y, z, w)
        r_pred = R.from_quat(predicted[i])

        r_error = r_pred.inv() * r_truth
        rotation_error_deg.append(r_error.magnitude() * (180 / np.pi))
    mean_rotation_error = np.mean(rotation_error_deg)
    return mean_rotation_error

def quat_to_rpy(quat):
    # quat: [x, y, z, w]
    r = R.from_quat(quat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def rpy_error(predicted, truth):
    try:
        predicted = predicted.values.tolist()
    except:
        pass
    try:
        truth = truth.values.tolist()
    except:
        pass
    roll, pitch, yaw = [], [], []
    for i in range(len(predicted)):
        rpy_pred = np.array(quat_to_rpy(predicted[i]))
        rpy_truth = np.array(quat_to_rpy(truth[i]))
        
        diff = (rpy_pred - rpy_truth + 180) % 360 - 180
        abs_errors = np.abs(diff)

        roll.append(abs_errors[0])
        pitch.append(abs_errors[1])
        yaw.append(abs_errors[2])
    
    return np.mean(roll), np.mean(pitch), np.mean(yaw)

def space(value, space=12, decimal=4):
    value = (int(value * 10**decimal)) / 10**decimal
    if value == 0:
        return ' '*(space - 6)
    elif len(str(value)) < 6:
        return ' '*(space - 6)
    else:
        space = space -  len(str(value))
        space = space if space > 0 else 0
    return ' '*space

def position_evaluation_overview(predicted, truth):
    r2_x, r2_y, r2_z = r2_score_xyz(truth=truth, predicted=predicted)
    mae_x, mae_y, mae_z = mae_xyz(truth=truth, predicted=predicted)
    euclidean = euclidean_distance(truth=truth, predicted=predicted)
    
    print("\nObject Poistion (X,Y,Z) Evaluation:")
    print("---------------------------------------------------------------")
    print(f"Euclidean           |                  {euclidean:.4f}                 |")
    print("---------------------------------------------------------------\n")
    print("---------------------------------------------------------------")
    print(f"Regression          | X           | Y           | Z           |")
    print("---------------------------------------------------------------")
    print(f"R² Score            | {r2_x:.4f}{space(r2_x)}| {r2_y:.4f}{space(r2_y)}| {r2_z:.4f}{space(r2_z)}|")
    print(f"Mean Absolute Error | {mae_x:.4f}{space(mae_x)}| {mae_y:.4f}{space(mae_y)}| {mae_z:.4f}{space(mae_z)}|")
    print("---------------------------------------------------------------\n")

def orientation_evaluation_overview(predicted, truth):
    quat_error = mean_quaternion_error(predicted=predicted, truth=truth)
    roll, pitch, yaw = rpy_error(predicted=predicted, truth=truth)
    
    print("Object Orientation (X,Y,Z,W) Evaluation:")
    print("------------------------------------------------------------------")
    print(f"Mean Quaternion Error  |                 {quat_error:.4f}                 |")
    print("------------------------------------------------------------------\n")
    print("------------------------------------------------------------------")
    print(f"3-Axis Rotation        | Roll        | Pitch       | Yaw         |")
    print("------------------------------------------------------------------")
    print(f"Mean Errors            | {roll:.4f}{space(roll)}| {pitch:.4f}{space(pitch)}| {yaw:.4f}{space(yaw)}|")
    print("------------------------------------------------------------------\n")

def class_evaluation(predicted, truth):
    print("Object Class Evaluation:")
    print("---------------------------------------------------------")
    print(classification_report(y_pred=predicted, y_true=truth))
    print("---------------------------------------------------------\n")
    
def progress_bar(progress, bar_size=50):
    progress = '='*int((progress*bar_size/100))
    space = ' '*int(bar_size - len(progress))
    bar = f'|{progress}{space}|'
    return bar
    
def dataset_inference(model, tokenizer, eval_dataset, size=None, system_message=None, batch_size=3, prompt_field='prompt', image_field='images', format=2, return_raw=False, temperature=1.5, min_p=0.1):
    results = [[],[],[]]
    raws = []
    eval_dataset.reset_index(inplace=True, drop=True)
    print(f'Inferencing on dataset of {len(eval_dataset)} records with batch size of {batch_size} per iteration:')
    size = size if size else (854,480)
    print(f'All Images will be resized to {size}')
    # sys.stdout.write(f"\rProgress: {0:.2f}%        {progress_bar(0)}")
    # sys.stdout.flush() 
    for index in range(0, len(eval_dataset), batch_size):
        if len(eval_dataset) < index + batch_size:
            prompt = eval_dataset[prompt_field][index:len(eval_dataset)]
            image = eval_dataset[image_field][index:len(eval_dataset)]
        else:
            prompt = eval_dataset[prompt_field][index:index+batch_size]
            image = eval_dataset[image_field][index:index+batch_size]
        
        pos, ori, obj, raw = batch_inference(model=model, tokenizer=tokenizer, size=size, system_message=system_message, prompts=prompt, image_paths=image, format=format, return_raw=return_raw, temperature=temperature, min_p=min_p)

        results[0].extend(pos)
        results[1].extend(ori)
        results[2].extend(obj)
        
        if return_raw:
            raws.extend(raw)
        
        prog = min(((index + batch_size)*100 / len(eval_dataset)), 100)
        sys.stdout.write(f"\rProgress: {prog:.2f}%        {progress_bar(prog)}")
        sys.stdout.flush()

    eval_dataset = eval_dataset.copy()    
    eval_dataset['pos_out'] = results[0]
    eval_dataset['ori_out'] = results[1]
    eval_dataset['obj_out'] = results[2]
    if return_raw:
        eval_dataset['raw_out'] = raws

    return eval_dataset

def validate_output_format(df, column_name):
    def check_parameter_counts(text):
        # Count occurrences of each tag
        obj_tags = re.findall(r"<obj>.*?</obj>", text)
        pose_tags = re.findall(r"<pose>.*?</pose>", text)
        orient_tags = re.findall(r"<orient>.*?</orient>", text)

        errors = []
        if len(obj_tags) != 1:
            errors.append(f"obj count = {len(obj_tags)}")
        if len(pose_tags) != 1:
            errors.append(f"pose count = {len(pose_tags)}")
        if len(orient_tags) != 1:
            errors.append(f"orient count = {len(orient_tags)}")

        return [] if not errors else errors
    
    def check_parameter_format(df, error_col):
        for i in range(len(df)):
            if isinstance(df['pos_out'][i], list):
                if len(df['pos_out'][i]) != 3:
                    df[error_col][i].append('incorrect position format')
            else:
                df[error_col][i].append('position is not listformat')

            if isinstance(df['ori_out'][i], list): 
                if len(df['ori_out'][i]) != 4:
                    df[error_col][i].append('incorrect orientation format')
            else:
                df[error_col][i].append('orientation is not list format')
    
    error_col = 'format_check'
    df[error_col] = df[column_name].apply(check_parameter_counts)
    check_parameter_format(df, error_col=error_col)

    return df

def get_error(df):
    df = df.copy()
    df = validate_output_format(df, 'raw_out')
    idx = df[df["format_check"].apply(len) > 0].index
    return df, idx

def eval_model(model, tokenizer, eval_dataset, size=None, system_message=None, batch_size=3, prompt_field='prompt', image_field='images', output_field='output', format=2, temperature=1.5, min_p=0.1):
    results = dataset_inference(model=model, tokenizer=tokenizer, eval_dataset=eval_dataset, size=size, system_message=system_message, batch_size=batch_size, prompt_field=prompt_field, image_field=image_field, format=format, return_raw=True, temperature=temperature, min_p=min_p)
  
    pos_truth = []
    ori_truth = []
    obj_truth = []
    print('\nExtracting true value from the output...')
    for i in range(len(eval_dataset)):
        pos, ori, obj = extract_position_and_orientation(text=eval_dataset[output_field][i], format=format)
        pos_truth.append(pos)
        ori_truth.append(ori)
        obj_truth.append(obj)

    results['pos_truth'] = pos_truth
    results['ori_truth'] = ori_truth
    results['obj_truth'] = obj_truth
    
    results, index = get_error(results)
    eval_results = results.drop(index)
    print(f'[INFO] {len(index)} records are excluded from the evaluation due to inconsistent output format')

    position_evaluation_overview(predicted=eval_results['pos_out'], truth=eval_results['pos_truth'])
    orientation_evaluation_overview(predicted=eval_results['ori_out'], truth=eval_results['ori_truth'])
    class_evaluation(predicted=eval_results['obj_out'], truth=eval_results['obj_truth'])
    
    return results, index