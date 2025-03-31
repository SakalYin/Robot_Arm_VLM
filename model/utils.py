from PIL import Image
import re
import json
from unsloth import FastVisionModel
from transformers import TextStreamer

def resize_image(image_input):
    if isinstance(image_input, str):  
        image = Image.open(image_input)
    else:  
        image = image_input

    image = image.resize((854, 480)).convert('RGB') 
    return image

def extract_position_and_orientation(text, format=2):
    """Extracts position and orientation data from JSON-like text output."""
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

def inference(image_path, prompt, model, tokenizer, system_message=None, temperature=1.5, min_p=0.1, format=2, return_value=True):
    """Performs inference and extracts position/orientation."""
    model = FastVisionModel.for_inference(model)  

    image = resize_image(image_path)
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
    generated_tokens = model.generate(
        **inputs,
        streamer=text_streamer,  
        max_new_tokens=128, 
        use_cache=True, 
        temperature=temperature, 
        min_p=min_p
    )

    # Decode output properly
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

    # Extract position and orientation
    position, orientation, object = extract_position_and_orientation(output_text, format=format)
    
    if return_value:
        return position, orientation, object

def batch_inference(image_paths, prompts, model, tokenizer, system_message=None, temperature=1.5, min_p=0.1, format=2, return_raw=False):
    """Batch inference with multiple images and prompts"""

    model = FastVisionModel.for_inference(model)  # put model into inference mode

    # Step 1: Preprocess images
    images = [resize_image(p) for p in image_paths]  # list of resized images

    # Step 2: Create messages (prompt list)
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
    ).to("cuda")

    # Step 4: Generate
    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=temperature,
        min_p=min_p
    )

    # Step 5: Decode batch outputs
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    
    position, orientation, object = [], [], []
    for item in outputs:
        pos, orient, obj = extract_position_and_orientation(item, format=format)
        position.append(pos)
        orientation.append(orient)
        object.append(obj)
    
    if return_raw:
        return position, orientation, object, outputs
    
    return position, orientation, object