from ._base import BaseGPTQForCausalLM, BaseGPTQForCausalLM_mixed_precision


class DeepSeekGPTQForCausalLM(BaseGPTQForCausalLM_mixed_precision):
    model_type = "deepseek"
    layer_type = "DeepseekDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    
    moe_1_list = []
    moe_2_list = []
    
    for part in ['gate_proj', 'up_proj']:
        for i in range(64):
            key = f"mlp.experts.{i}.{part}"
            moe_1_list.append(key)

    for part in ['gate_proj', 'up_proj']:
        key = f"mlp.shared_experts.{part}"
        moe_1_list.append(key)
        # mlp.shared_experts.gate_proj
        
    for i in range(64):
        for part in ['down_proj']:
            key = f"mlp.experts.{i}.{part}"
            moe_2_list.append(key)
            
    for part in ['down_proj']:
        key = f"mlp.shared_experts.{part}"
        moe_2_list.append(key)


    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        moe_1_list,
        moe_2_list
    ]
    
    normal_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"], 
        ["mlp.down_proj"],
    ]


__all__ = ["DeepSeekGPTQForCausalLM"]
