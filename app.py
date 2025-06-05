import unsloth
from io import BytesIO
from flask import Flask, request, jsonify
import os
from datetime import datetime
from model.qwenvl.utils import inference
from unsloth import FastVisionModel
from PIL import Image

app = Flask(__name__)

model, processor = FastVisionModel.from_pretrained(
    r'checkpoints\Qwen2b-Robot-Arm-Unsloth\7B\V2\P3\checkpoint-440',
    load_in_4bit=True,  
    use_gradient_checkpointing="unsloth", 
    local_files_only=True,
    attn_implementation="flash_attention_2",
)

# directory = 'D:/Sakal/AI_FARM/Robot_Arm_Model/tmp/flask_img' # Save images to this directory or None to disable saving
directory = None

if directory is not None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Folder created: {directory}")

system_message = 'You are a Visual Language Model Trained to output robot arm end-effectors parameters.' \
'Base on the user requests, locate the appropriate object in the image and you must return the position and orientation to reach it in xml format.'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # request images and prompt
        file = request.files['image']
        prompt = request.form.get('prompt')
        image = Image.open(BytesIO(file.read()))
        
        # save image
        if directory is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(directory, f"image_{timestamp}.png")
            image.save(filename)

        pos, ori, obj = inference(image_path = image, prompt = prompt, system_message = system_message, model = model, tokenizer = processor)
        
        return jsonify({'status': 'success', 'prediction': [pos, ori, obj]})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)