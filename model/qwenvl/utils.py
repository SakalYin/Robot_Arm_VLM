from PIL import Image
import re
import json
from unsloth import FastVisionModel
from transformers import TextStreamer
import pandas as pd

def resize_image(image_input, size=None):
    """Load image from path or matrices and resize them"""
    size = size if size else (854,480)
    if isinstance(image_input, str):  
        image = Image.open(image_input)
    else:  
        image = image_input

    size = size if size else (854,480)
    image = image.resize(size).convert('RGB') 
    return image

def extract_position_and_orientation(text, format=2):
    """Extracts object class, position and orientation data from JSON-like text output."""
    if format == 1:
        match = re.search(r'\[.*\]', text)
        if match:
            json_data = match.group(0)  
            try:
                data = json.loads(json_data)  
                if data:
                    position = data[0].get("position")
                    orientation = data[0].get("orientation")
                    return position, orientation
            except json.JSONDecodeError:
                print("Error decoding JSON:", json_data)
        return None, None
    
    if format == 2:
        pose_match = re.search(r"<pose>([-0-9.,]+)</pose>", text)
        orient_match = re.search(r"<orient>([-0-9.,]+)</orient>", text)
        object_match = re.search(r"<obj>(.*?)</obj>", text)
        
        if pose_match and orient_match and object_match:
            position = pose_match.group(1).split(',')
            orientation = orient_match.group(1).split(',')
            object = object_match.group(1)
            return position, orientation, object
        else:
            return None, None, None

def inference(image_path, prompt, model, tokenizer, size=None, system_message=None, temperature=1.5, min_p=0.1, sample=True, format=2, return_value=True):
    """Performs inference and extracts position/orientation."""
    model = FastVisionModel.for_inference(model)  

    image = resize_image(image_path, size=size)
    if system_message is None:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message if system_message else "You are a helpful assistant."}],
            },
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}]
            }
        ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate text without using the streamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    if sample:
        generated_tokens = model.generate(
            **inputs,
            streamer=text_streamer,  
            max_new_tokens=128, 
            use_cache=True, 
            temperature=temperature, 
            min_p=min_p,
        )
    else:
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
        )

    # Decode output properly
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

    # Extract position and orientation
    position, orientation, object = extract_position_and_orientation(output_text, format=format)
    
    if return_value:
        return position, orientation, object

def batch_inference(image_paths, prompts, model, tokenizer, size=None, system_message=None, temperature=1.5, min_p=0.1, format=2, return_raw=False):
    """Batch inference with multiple images and prompts"""

    model = FastVisionModel.for_inference(model)  # put model into inference mode

    # Step 1: Preprocess images
    images = [[resize_image(p, size=size)] for p in image_paths]  # list of resized images

    # Step 2: Create messages (prompt list)
    if system_message is None:
        messages_list = [
        [{
            "role": "user", 
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]
        for prompt in prompts
    ]
    else:
        messages_list = [
            [{
                "role": "system",
                "content": [{"type": "text", "text": system_message if system_message else "You are a helpful assistant."}],
            },
            {
                "role": "user", 
                "content": [{"type": "image"}, {"type": "text", "text": prompt}]
            }]
            for prompt in prompts
        ]

    # Step 3: Tokenize (this is where batching happens)
    input_texts = [tokenizer.apply_chat_template(msgs, add_generation_prompt=True) for msgs in messages_list]

    inputs = tokenizer(
        images,                  # list of images
        input_texts,             # list of prompts
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,            # important for batching
        padding_side="left",       # pad on the left side
    ).to("cuda")
    # input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    # Step 4: Generate
    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )

    # Step 5: Decode batch outputs
    raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    raw = [output.strip() for output in raw]
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_tokens)
    ]
    outputs = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # outputs = []
    # for gen_ids, input_len in zip(generated_tokens, input_lengths):
    #     gen_only = gen_ids[input_len:]  # Remove input tokens
    #     decoded = tokenizer.decode(gen_only, skip_special_tokens=True)
    #     outputs.append(decoded.strip())
    
    position, orientation, object = [], [], []
    for item in outputs:
        pos, orient, obj = extract_position_and_orientation(item, format=format)
        position.append(pos)
        orientation.append(orient)
        object.append(obj)
    
    if return_raw:
        return position, orientation, object, outputs, raw
    
    return position, orientation, object

def verify_ouput(outputs):
    """Verify format error for a list of outputs"""
    results = []

    for i, output in enumerate(outputs):
        errors = []

        obj_match = re.findall(r"<obj>(.*?)</obj>", output)
        pose_match = re.findall(r"<pose>(.*?)</pose>", output)
        orient_match = re.findall(r"<orient>(.*?)</orient>", output)

        if (len(obj_match) > 1) or (len(pose_match) > 1) or (len(orient_match) > 1):
            errors.append("Repetitions")
            continue
        elif (len(obj_match) == 0) or (len(pose_match) == 0) or (len(orient_match) == 0):
            errors.append("Missing Parameters")
            continue
        else:
            position = pose_match[0].split(',')
            orientation = orient_match[0].split(',')
            
            if len(position) != 3:
                errors.append("Position format")
            else:
                try:
                    position = [float(p) for p in position]
                except:
                    errors.append("Position float format")
            if len(orientation) != 4:
                errors.append("Orientation format")
            else:
                try:
                    orientation = [float(o) for o in orientation]
                except:
                    errors.append("Orientation float format")
        results.append([i, output, errors])

    return pd.DataFrame(results, columns=['index', 'raw', 'errors'])