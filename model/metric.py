import numpy as np
from model.qwenvl.utils import extract_position_and_orientation

def euclidean_distance(predicted, truth):
    distances = [abs(((float(p1[0]) - float(p2[0]))**2 + (float(p1[1]) - float(p2[1]))**2 + (float(p1[2]) - float(p2[2]))**2) ** 0.5)
                    for p1,p2 in zip(predicted, truth)]
    return  np.mean(distances)

def mae(predicted, truth):
    ae = [abs(float(p1) - float(p2)) for p1,p2 in zip(predicted, truth)]
    mae = np.mean(ae)
    return mae

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    print(logits.shape)
    print(labels.shape)
    # Convert logits to predicted token ids (argmax to get the predicted token)
    predictions = logits.argmax(axis=-1)
    predicted_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True, padding_side='left')
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True, padding_side='left')
    # extract values (fix to use batch)
    predicted_param = [[pos,ori] for text in predicted_texts for pos, ori,obj in extract_position_and_orientation(text)]
    label_param = [[pos,ori] for text in labels_texts for pos, ori,obj in extract_position_and_orientation(text)]
    # transpose
    predict_param = np.array([predicted_param]).T.astype(torch.bfloat16)
    label_param = np.array([label_param]).T.astype(torch.bfloat16)
    #euclidean
    distances = euclidean_distance(predict_param[0], label_param[0])

    predicted_xyz, truth_xyz = np.array(predict_param).T, np.array(label_param).T
    mae_x = mae(predicted_xyz[0], truth_xyz[0])
    mae_y = mae(predicted_xyz[1], truth_xyz[1])
    mae_z = mae(predicted_xyz[2], truth_xyz[2])

    return {
        'euclidean': distances,
        'mea_x': mea_x,
        'mea_y': mea_y,
        'mea_z': mea_z
        }