# backbone/llama_lora.py
"""
Loads LLaMA backbone (optionally with LoRA adapters).

Returns:
    peft_model : PEFT-wrapped model for saving/loading  (None in inference mode)
    llama      : the inner LlamaModel that accepts `inputs_embeds`
"""
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_llama_with_lora(model_id: str, lora_config, dtype):
    """
    Args:
        model_id    : HuggingFace model id
        lora_config : peft.LoraConfig  or  None (inference / load-from-disk mode)
        dtype       : torch dtype (bfloat16 recommended)

    Returns:
        peft_model  : PEFT-wrapped causal LM  (None when lora_config is None)
        llama       : inner LlamaModel  (accepts inputs_embeds)
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    
    if lora_config is None:
        # Inference mode: return base model directly
        return None, base_model.model
    
    # Training mode: wrap with PEFT
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()  # Log trainable params for sanity check
    
    #Freeze everything except LoRA adapters
    for name, param in peft_model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    
    # Inner LlamaModel (accepts inputs_embeds)
    llama = peft_model.model.model
    
    return peft_model, llama 