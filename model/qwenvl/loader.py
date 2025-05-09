from unsloth import FastVisionModel

def load_model(model_path, load_in_4bit=True, local_files_only=True):
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,  
        use_gradient_checkpointing="unsloth",  
        local_files_only=local_files_only,
        attn_implementation="flash_attention_2",
    )
    return model, processor


