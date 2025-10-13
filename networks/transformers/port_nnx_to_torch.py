import numpy as np
import torch as th

def transpose_linear(flax_kernel):
    """
    Given a Flax linear kernel of shape (in_features, out_features),
    return the equivalent PyTorch weight of shape (out_features, in_features).
    """
    return th.from_numpy(np.array(flax_kernel).T)

def transpose_conv(flax_kernel):
    """
    Given a Flax convolution kernel of shape (kh, kw, in_channels, out_channels),
    return the equivalent PyTorch weight of shape (out_channels, in_channels, kh, kw).
    """
    return th.from_numpy(np.array(flax_kernel).transpose(3, 2, 0, 1))

def combine_qkv(query, key, value):
    """
    Given three Flax kernels for query, key, and value (each with shape e.g. (in_features, out_features)
    - note that in Flax these might be stored as (in_features, head_dim, num_heads) so you may need
    to reshape them into (in_features, hidden_dim) first),
    combine and return a kernel with shape (in_features, out_features*3) and then transpose it.
    """
    # Here we assume each is stored as (in_features, hidden_dim) with hidden_dim = 768.
    hidden_dim = query.shape[0]
    q = np.array(query).reshape(hidden_dim, -1).T
    k = np.array(key).reshape(hidden_dim, -1).T
    v = np.array(value).reshape(hidden_dim, -1).T
    combined = np.concatenate([q, k, v], axis=0)  # shape (in_features, 2304)
    return th.from_numpy(combined)  # PyTorch expects (2304, in_features)

def combine_qkv_bias(q_bias, k_bias, v_bias):
    q = np.array(q_bias).reshape(-1)
    k = np.array(k_bias).reshape(-1)
    v = np.array(v_bias).reshape(-1)
    return th.from_numpy(np.concatenate([q, k, v], axis=0))

def convert_x_embedder(flax_params):
    """
    Convert the Flax x_proj (the convolution for patch embedding)
    to PyTorch's x_embedder.proj.
    """
    torch_state = {}
    flax_x_proj = flax_params["x_proj"]
    torch_state["x_embedder.proj.weight"] = transpose_conv(flax_x_proj["kernel"])
    torch_state["x_embedder.proj.bias"] = th.from_numpy(np.array(flax_x_proj["bias"]))
    return torch_state

def convert_y_embedder(flax_params):
    """
    Convert the Flax label embedder (y_embedder) to PyTorch.
    (Assumes the embedding table key is named 'embedding_table'.)
    """
    torch_state = {}
    flax_y_embed = flax_params["y_embedder"]
    torch_state["y_embedder.embedding_table.weight"] = th.from_numpy(np.array(flax_y_embed["embedding_table"]["embedding"]))
    return torch_state

def convert_t_embedder(flax_params):
    """
    Convert the Flax time embedder.
    Here the Flax version uses a Sequential with two Linear layers
    (and an initial 'gaussian_basis' vector). The corresponding PyTorch module
    (TimestepEmbedder) has a sequential with Linear, SiLU, Linear.
    
    (Note: In our printed definitions the first linear in Flax has shape (512,768)
    while the torch linear expects (256,768). You may need to resolve this difference.)
    For illustration we assume the conversion is direct and only applies a transpose.
    """
    torch_state = {}
    flax_t_proj = flax_params["t_embedder"]["mlp"]
    # Convert first linear layer:
    torch_state["t_embedder.mlp.0.weight"] = transpose_linear(flax_t_proj["layers"][0]["kernel"])
    torch_state["t_embedder.mlp.0.bias"] = th.from_numpy(np.array(flax_t_proj["layers"][0]["bias"]))
    # The activation (SiLU) is stateless.
    # Convert second linear layer:
    torch_state["t_embedder.mlp.2.weight"] = transpose_linear(flax_t_proj["layers"][2]["kernel"])
    torch_state["t_embedder.mlp.2.bias"] = th.from_numpy(np.array(flax_t_proj["layers"][2]["bias"]))
    return torch_state

def convert_block(flax_block, block_idx):
    """
    Convert one DiT block from Flax to PyTorch.
    This function converts:
      - The attention module: combining separate query, key, value linear layers into a single qkv linear,
        and converting the output projection.
      - The MLP: converting the two linear layers.
      - The adaLN module: converting the linear layer inside the sequential.
    We assume that any LayerNorm modules that do not have learnable parameters need no conversion.
    """
    torch_state = {}
    prefix = f"blocks.{block_idx}"
    
    # --- Attention ---
    flax_attn = flax_block["attn"]
    # Combine q, k, v weights and biases:
    torch_state[f"{prefix}.attn.qkv.weight"] = combine_qkv(
        flax_attn["query"]["kernel"],
        flax_attn["key"]["kernel"],
        flax_attn["value"]["kernel"]
    )
    torch_state[f"{prefix}.attn.qkv.bias"] = combine_qkv_bias(
        flax_attn["query"]["bias"],
        flax_attn["key"]["bias"],
        flax_attn["value"]["bias"]
    )
    # Convert the output projection (assumed to be under key 'out')
    flax_proj = flax_attn["out"]
    # If flax_proj["kernel"] is stored as (num_heads, head_dim, out_features) or flattened,
    # reshape accordingly; here we assume it is stored as (hidden_dim, out_features)
    torch_state[f"{prefix}.attn.proj.weight"] = transpose_linear(flax_proj["kernel"].reshape(-1, 768))
    torch_state[f"{prefix}.attn.proj.bias"] = th.from_numpy(np.array(flax_proj["bias"].reshape(-1)))
    
    # --- MLP ---
    flax_mlp = flax_block["mlp"]
    torch_state[f"{prefix}.mlp.fc1.weight"] = transpose_linear(flax_mlp["linear1"]["kernel"])
    torch_state[f"{prefix}.mlp.fc1.bias"] = th.from_numpy(np.array(flax_mlp["linear1"]["bias"]))
    torch_state[f"{prefix}.mlp.fc2.weight"] = transpose_linear(flax_mlp["linear2"]["kernel"])
    torch_state[f"{prefix}.mlp.fc2.bias"] = th.from_numpy(np.array(flax_mlp["linear2"]["bias"]))
    
    # --- adaLN modulation ---
    # In Flax, this is a Sequential with a silu function and then a Linear layer.
    # In PyTorch it is named adaLN_modulation.
    flax_adamod = flax_block["adaLN_mod"]["layers"][1]  # the linear layer
    torch_state[f"{prefix}.adaLN_modulation.1.weight"] = transpose_linear(flax_adamod["kernel"])
    torch_state[f"{prefix}.adaLN_modulation.1.bias"] = th.from_numpy(np.array(flax_adamod["bias"]))
    
    return torch_state

def convert_final_layer(flax_final):
    """
    Convert the final layer from Flax to PyTorch.
    """
    torch_state = {}
    # If the final norm has parameters, convert them (here we assume none)
    # Convert the final linear layer:
    torch_state["final_layer.linear.weight"] = transpose_linear(flax_final["linear"]["kernel"])
    torch_state["final_layer.linear.bias"] = th.from_numpy(np.array(flax_final["linear"]["bias"]))
    # Convert the final adaLN modulation layer:
    flax_final_adamod = flax_final["adaLN_mod"]["layers"][1]
    torch_state["final_layer.adaLN_modulation.1.weight"] = transpose_linear(flax_final_adamod["kernel"])
    torch_state["final_layer.adaLN_modulation.1.bias"] = th.from_numpy(np.array(flax_final_adamod["bias"]))
    return torch_state

def convert_flax_to_torch(flax_state):
    """
    Given a nested Flax state dictionary (for DiT), convert it into a dictionary
    that can be loaded as the state_dict for the PyTorch DiT model.
    
    IMPORTANT:
      • This function assumes that the Flax state dictionary has keys:
          "x_proj", "t_embedder", "y_embedder", "blocks", "final_layer"
      • It applies transpositions and weight combinations where needed.
      • You will likely need to adjust key names and shapes to match your exact implementations.
    """
    torch_state = {}
    torch_state.update({'pos_embed': th.from_numpy(np.array(flax_state['x_embedder']['pe']))})
    torch_state.update(convert_x_embedder(flax_state))
    torch_state.update(convert_t_embedder(flax_state))
    torch_state.update(convert_y_embedder(flax_state))
    
    for i, flax_block in flax_state["blocks"].items():
        torch_state.update(convert_block(flax_block, i))
        
    torch_state.update(convert_final_layer(flax_state["final_layer"]))
    return torch_state

# === Example usage ===
# Assuming you have loaded a Flax checkpoint (e.g. as a nested dict called `flax_state`)
# and you have a PyTorch model (e.g. `torch_model`), you could convert and load the weights like:

# torch_state = convert_flax_to_torch(flax_state)
# torch_model.load_state_dict(torch_state)

# Note: In practice you may need to do additional debugging, adjust for mismatched dimensions,
# and verify that every parameter is converted correctly.
