import numpy as np
from sklearn.metrics import classification_report
from scipy.spatial.transform import Rotation as R
import sys
from model.utils import batch_inference, extract_position_and_orientation

def euclidean_distance(predicted, truth, summarize='mean'):
    distances = [abs(((float(p1[0]) - float(p2[0]))**2 + (float(p1[1]) - float(p2[1]))**2 + (float(p1[2]) - float(p2[2]))**2) ** 0.5)
                    for p1,p2 in zip(predicted, truth)]
    
    if summarize == 'mean':
        distances = np.mean(distances)
    
    return distances

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
    
def dataset_inference(model, tokenizer, eval_dataset, system_message=None, batch_size=3, prompt_field='prompt', image_field='images', format=2, return_raw=False):
    results = [[],[],[]]
    raws = []
    eval_dataset.reset_index(inplace=True, drop=True)
    print(f'Inferencing on dataset of {len(eval_dataset)} records with batch size of {batch_size} per iteration:')
    # sys.stdout.write(f"\rProgress: {0:.2f}%        {progress_bar(0)}")
    # sys.stdout.flush() 
    for index in range(0, len(eval_dataset), batch_size):
        if len(eval_dataset) < index + batch_size:
            prompt = eval_dataset[prompt_field][index:len(eval_dataset)]
            image = eval_dataset[image_field][index:len(eval_dataset)]
        else:
            prompt = eval_dataset[prompt_field][index:index+batch_size]
            image = eval_dataset[image_field][index:index+batch_size]
        
        if return_raw:
            pos, ori, obj, raw = batch_inference(model=model, tokenizer=tokenizer, system_message=system_message, prompts=prompt, image_paths=image, format=format, return_raw=return_raw)
        else:
            pos, ori, obj = batch_inference(model=model, tokenizer=tokenizer, prompts=prompt, image_paths=image, format=format, return_raw=return_raw)
        results[0].extend(pos)
        results[1].extend(ori)
        results[2].extend(obj)
        
        if return_raw:
            raws.extend(raw)
        
        prog = min(((index + batch_size)*100 / len(eval_dataset)), 100)
        sys.stdout.write(f"\rProgress: {prog:.2f}%        {progress_bar(prog)}")
        sys.stdout.flush() 
        
    if return_raw:
        return results, raws
    return results

def get_error(lst, x=None):
    return [i for i, value in enumerate(lst) if value == x]

def remove_error(lst, index):
    return [lst[i] for i in range(len(lst)) if i not in index]

def eval_model(model, tokenizer, eval_dataset, system_message=None, batch_size=3, prompt_field='prompt', image_field='images', output_field='output', format=2, return_raw=False):
    if return_raw:
        results, raw = dataset_inference(model=model, tokenizer=tokenizer, eval_dataset=eval_dataset, batch_size=batch_size, prompt_field=prompt_field, image_field=image_field, format=format, return_raw=return_raw)
    else:
        results = dataset_inference(model=model, tokenizer=tokenizer, eval_dataset=eval_dataset, system_message=system_message, batch_size=batch_size, prompt_field=prompt_field, image_field=image_field, format=format, return_raw=return_raw)
        
    pos_truth = []
    ori_truth = []
    obj_truth = []
    print('\nExtracting true value from the output...')
    for i in range(len(eval_dataset)):
        pos, ori, obj = extract_position_and_orientation(text=eval_dataset[output_field][i], format=format)
        pos_truth.append(pos)
        ori_truth.append(ori)
        obj_truth.append(obj)
    
    index = get_error(lst=results[0])
    results[0], results[1], results[2] = remove_error(results[0], index), remove_error(results[1], index), remove_error(results[2], index)
    pos_truth, ori_truth, obj_truth = remove_error(pos_truth, index), remove_error(ori_truth, index), remove_error(obj_truth, index)
    
    position_evaluation_overview(predicted=results[0], truth=pos_truth)
    orientation_evaluation_overview(predicted=results[1], truth=ori_truth)
    class_evaluation(predicted=results[2], truth=obj_truth)
    
    if return_raw:
        return results, [pos_truth, ori_truth, obj_truth], raw, index
    return results, [pos_truth, ori_truth, obj_truth]