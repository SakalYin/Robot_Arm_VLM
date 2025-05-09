# Fine-Tuning Visual Language Model for Robotic Arms
<hr style="margin: 10;" />

This project focuses on fine-tuning visual language models to predict the position and orientation of a robotic arm for reaching a target object based on a human command and an image from a camera view. The system interprets natural language instructions and visual context to generate actionable pose outputs. Models trained so far include Qwen-VL (2B and 7B), LLaVA-NeXT 7B, and Llama Vision 11B.


![alt text](<tmp/readme_img/Screenshot from 2025-05-08 16-28-04.png>)


## Installation

1. **Clone the repository** (if not already cloned):

   ```bash
   git clone git@gitlab.aifarm.dev:factoryai-visionai/llm-for-human-robot-interaction/robot_arm_vlm.git
   ```

2. **Libraries and Requirements**:
   Python version 3.10, torch 2.60 + CUDA
   ```bash
    pip install -r requirement.txt
   ```

   For Windows only, install triton and flash_attn with:
   ```bash
    pip install triton-windows==3.2.0.post19
    pip install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4+cu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
   ```


## Data Structure & Preprocessing
#### **Data Structure**
The dataset is organized by scenarios, each containing images and corresponding labels for robotic arm training.

    Current repo/
    └── Robot_Arm_Data/
        ├── Scenario_1/
        │ ├── eff_label.txt 
        │ ├── label.txt
        │ ├── img_1.png
        │ ├── img_2.png
        │ └── ...
        ├── Scenario_2/
        │ ├── eff_label.txt
        │ ├── label.txt
        │ ├── img_1.png
        │ ├── img_2.png
        │ └── ...
        └── ...
- **Scenario** refers to the different simulation environment setup such as camera locations, lightings, table and ground floor types
- **eff_label.txt** contains the actual robot arm position and orient during the capture of each images (used for identifying and clean out cases where robot arm doesn't move resulting in static duplicate images)
- **label.txt** contains the position and orientation for the robot arm to reach the target object and the target object name in each corresponding images

It's **optional** to separate different scenarios into folders

#### **Data Preprocessing**
Use the function **prepare_data()** to read and clean dataset and pair with random prompts into a json format:

   ```bash
    from model.data_prep import prepare_data
    prepare_data('Robot_Arm_Data/folder', drop_static=True, float_round=3)
   ```

A json file contains the image paths, target orientation/position and corresponding prompt will appear in **Robot_Arm_Data** folder under the same name as the preprocessed folder. Combine then split train/test as in **data_prep.ipynb**

Sample from train.json:

   ```bash
    {
        "images":"Robot_Arm_Data\/bright_default_cam\/image_20250324_152040_2.png",
        "prompt":"This meal would taste better with a peach on the side.",
        "output":"<obj>peach<\/obj> <pose>0.88,-0.475,0.63<\/pose> <orient>0.4468,0.8508,0.2185,0.147<\/orient>",
    }
   ```

## Model Training
Code for training different models are provided in the notebooks came with the repo.

## Quick Inferencing
Inference with **inference()** (for unsloth VLM only)

   ```bash
    from model.qwenvl.loader import load_model
    from model.qwenvl.utils import inference

    model, processor = load_model('model or checkpoint_path')
    system_message = None # (optional)

    inference(model=model, tokenizer=processor, temperature=1.5, min_p=0.1, return_value=False,
              system_message=system_message,
              image_path='Robot_Arm_Data/dark_default_cam/image_20250325_222612_1.png',
              size=(854, 480),
              prompt='Something cold and fizzy would go well with my meal.')
   ```

   Or use **batch_inference()** for batch inference by passing multiple images and prompt in list format.

## Model Evaluation
Evaluate model on a test dataset using eval_model() function:

   ```bash
    from model.evaluation import eval_model
    results, error_index = eval_model(model=model, tokenizer=processor, system_message=system_message,
                           size=(854,480), eval_dataset=data, batch_size=3, return_raw=True)
   ```

The function will invoke predictions, remove predictions with format errors, evaluate and print the evaluation tables using metric such as euclidean distance, mse, orientation errors and classification_report. Refer to [**Model_Evaluation.ipynb**](Model_Evaluation.ipynb).

Results from model trained with this repos config and dataset are as follow:

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Euclidean</th>
    <th>X MSE</th>
    <th>Y MSE</th>
    <th>Z MSE</th>
    <th>Roll Error</th>
    <th>Pitch Error</th>
    <th>Yaw Error</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>Qwen2VL-2B  <sup><sup>(Trained with Camera Location)</sup></sup></td>
    <td>0.1089</td>
    <td>0.0505</td>
    <td>0.0575</td>
    <td>0.0363</td>
    <td>6.8334</td>
    <td>5.4204</td>
    <td>6.7619</td>
  </tr>
  <tr>
    <td>Qwen2VL-2B</td>
    <td>0.3865</td>
    <td>0.1239</sup></sup></td>
    <td>0.3411</td>
    <td>0.0000</td>
    <td>2.1462</td>
    <td>0.9029</td>
    <td>1.1926</td>
  </tr>
  <tr>
    <td>Qwen2VL-7B</td>
    <td>0.0509</td>
    <td>0.0305</td>
    <td>0.0320</td>
    <td>0.0008</td>
    <td>0.4148</td>
    <td>0.1066</td>
    <td>0.1734</td>
  </tr>
  <tr>
    <td>LlaVA-NeXT 7B</td>
    <td>0.0480</td>
    <td>0.0306</td>
    <td>0.0296</td>
    <td>0.0000</td>
    <td>0.0000</td>
    <td>0.0000</td>
    <td>0.0000</td>
  </tr>
  <tr>
    <td>Llama Vision 11B</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

Model Checkpoint Collections : [Collections](https://huggingface.co/collections/SakalYin/robotarm-eff-single-action-681daa76a9286b53a51b401d)

## API
**app.py** will load model from path specified inside the code, wait for request (prompt, images), save the image inside **tmp** then invoke model prediction to return [target_object, position, orientation]

Example Request:

   ```bash
    import requests
    import cv2

    model_ip = 'your_ip'
    model_port = 5000
    URL = f'http://{model_ip}:{model_port}/predict'

    image = cv2.imread(image_path) # resize if needed
    _, img_encoded = cv2.imencode('.png', image)
    
    files = {'image': img_encoded.tobytes()}
    data = {'prompt': prompt}
    
    print('Sending input and Waiting for response.....')
    response = requests.post(URL, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        pos, ori = result['prediction'][0], result['prediction'][1]
        print(f'Recieved Position {pos}, Orientation {ori}')
   ```



