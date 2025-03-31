from unsloth import FastVisionModel


def load_model(model_path):
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,  
        use_gradient_checkpointing="unsloth",  
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    return model, processor


