import numpy as np
import torch as th

def th_to_jax(tensor):
    return tensor.cpu().numpy().astype(np.float32)

def transpose_linear(th_kernel):
    """
    Given a torch linear kernel of shape (out_features, in_features).
    return the equivalent Flax weight of shape (in_features, out_features),
    """
    return th_to_jax(th_kernel.T)

def transpose_conv(th_kernel):
    """
    Given a PyTorch convolution kernel of shape (out_channels, in_channels, kh, kw).
    return the equivalent Flax weight of shape (kh, kw, in_channels, out_channels),
    """
    return th_to_jax(th_kernel).transpose(2, 3, 1, 0)

def unbind_qkv(th_weight, num_heads, head_dim):
    """
    Given the combined PyTorch qkv kernel of shape (out_features*3, in_features)
    unbind and return three Flax Kernel with shape (in_features, out_features)
    """
    # Here we assume each is stored as (in_features, hidden_dim) with hidden_dim = 768.
    qw, kw, vw = th_weight.chunk(3, dim=0)
    qw, kw, vw = (
        qw.T.reshape(-1, num_heads, head_dim),
        kw.T.reshape(-1, num_heads, head_dim),
        vw.T.reshape(-1, num_heads, head_dim),
    )
    return th_to_jax(qw), th_to_jax(kw), th_to_jax(vw)

def unbind_qkv_bias(th_bias, num_heads, head_dim):
    qb, kb, vb = th_bias.chunk(3, dim=0)
    qb, kb, vb = qb.reshape(num_heads, head_dim), kb.reshape(num_heads, head_dim), vb.reshape(num_heads, head_dim)
    return th_to_jax(qb), th_to_jax(kb), th_to_jax(vb)

def convert_x_embedder(th_params):
    """
    Convert the Flax x_proj (the convolution for patch embedding)
    to PyTorch's x_embedder.proj.
    """
    flax_params = {}
    flax_params["kernel"] = transpose_conv(th_params["x_embedder.proj.weight"])
    flax_params["bias"] = th_to_jax(th_params["x_embedder.proj.bias"])
    return flax_params

def convert_y_embedder(th_params):
    """
    Convert the Flax label embedder (y_embedder) to PyTorch.
    (Assumes the embedding table key is named 'embedding_table'.)
    """

    flax_params = {}
    flax_params["embedding_table"] = {}
    flax_params["embedding_table"]["embedding"] = th_to_jax(th_params["y_embedder.embedding_table.weight"])

    return flax_params

def convert_t_embedder(th_params):
    """
    Convert the Flax time embedder.
    Here the Flax version uses a Sequential with two Linear layers
    (and an initial 'gaussian_basis' vector). The corresponding PyTorch module
    (TimestepEmbedder) has a sequential with Linear, SiLU, Linear.
    
    (Note: In our printed definitions the first linear in Flax has shape (512,768)
    while the torch linear expects (256,768). You may need to resolve this difference.)
    For illustration we assume the conversion is direct and only applies a transpose.
    """

    flax_params = {}
    flax_params["mlp"] = {}
    flax_params["mlp"]["layers"] = {
        0: {
            "kernel": transpose_linear(th_params["t_embedder.mlp.0.weight"]),
            "bias": th_to_jax(th_params["t_embedder.mlp.0.bias"]),
        },
        2: {
            "kernel": transpose_linear(th_params["t_embedder.mlp.2.weight"]),
            "bias": th_to_jax(th_params["t_embedder.mlp.2.bias"]),
        }
    }
    flax_params["gaussian_basis"] = th_to_jax(th_params["t_embedder.W"])

    return flax_params

def convert_block(th_params, block_idx):
    """
    Convert one DiT block from PyTorch to Flax.
    This function converts:
      - The attention module: combining separate query, key, value linear layers into a single qkv linear,
        and converting the output projection.
      - The MLP: converting the two linear layers.
      - The adaLN module: converting the linear layer inside the sequential.
    We assume that any LayerNorm modules that do not have learnable parameters need no conversion.
    """
    flax_params = {}
    prefix = f"blocks.{block_idx}"
    flax_params["attn"] = {}

    flax_params["attn"]["to_qkv"] = {
        "kernel": transpose_linear(th_params[f"{prefix}.attn.qkv.weight"]),
        "bias": th_to_jax(th_params[f"{prefix}.attn.qkv.bias"])
    }

    flax_params["attn"]["out"] = {
        "kernel": transpose_linear(th_params[f"{prefix}.attn.proj.weight"]),
        "bias": th_to_jax(th_params[f"{prefix}.attn.proj.bias"])
    }

    flax_params["mlp"] = {
        "linear12": {
            "kernel": transpose_linear(th_params[f"{prefix}.mlp.w12.weight"]),
            "bias": th_to_jax(th_params[f"{prefix}.mlp.w12.bias"])
        },
        "linear3": {
            "kernel": transpose_linear(th_params[f"{prefix}.mlp.w3.weight"]),
            "bias": th_to_jax(th_params[f"{prefix}.mlp.w3.bias"])
        }
    }

    flax_params["norm1"] = {
        "rms_weight": th_to_jax(th_params[f"{prefix}.norm1.weight"]),
    }
    flax_params["norm2"] = {
        "rms_weight": th_to_jax(th_params[f"{prefix}.norm2.weight"]),
    }

    flax_params["adaLN_mod"] = {
        "layers": {
            1: {
                "kernel": transpose_linear(th_params[f"{prefix}.adaLN_modulation.1.weight"]),
                "bias": th_to_jax(th_params[f"{prefix}.adaLN_modulation.1.bias"])
            }
        }
    }

    return flax_params

def convert_final_layer(th_final):
    """
    Convert the final layer from Flax to PyTorch.
    """

    flax_params = {}
    flax_params["linear"] = {
        "kernel": transpose_linear(th_final["final_layer.linear.weight"]),
        "bias": th_to_jax(th_final["final_layer.linear.bias"])
    }

    flax_params["norm"] = {
        "rms_weight": th_to_jax(th_final["final_layer.norm_final.weight"]),
    }

    flax_params["adaLN_mod"] = {
        "layers": {
            1: {
                "kernel": transpose_linear(th_final["final_layer.adaLN_modulation.1.weight"]),
                "bias": th_to_jax(th_final["final_layer.adaLN_modulation.1.bias"])
            }
        }
    }
    return flax_params

def convert_torch_to_flax(th_state, depth, encoder_depth):
    """
    Given a flattend torch state dictionary (for DiT), convert it into a nested dictionary
    that can be loaded as the state for the Flax NNX model.
    
    IMPORTANT:
      • This function assumes that the Flax state dictionary has keys:
          "x_proj", "t_embedder", "y_embedder", "blocks", "final_layer"
      • It applies transpositions and weight combinations where needed.
      • You will likely need to adjust key names and shapes to match your exact implementations.
    """
    flax_params = {}

    flax_params['x_proj'] = convert_x_embedder(th_state)
    flax_params["s_proj"] = {
        'kernel': transpose_conv(th_state['s_embedder.proj.weight']),
        'bias': th_to_jax(th_state['s_embedder.proj.bias']),
    }
    flax_params['s_embedder'] = {
        'pe': th_to_jax(th_state['pos_embed']),
    }
    flax_params["s_projector"] = {
        "kernel": transpose_linear(th_state['s_projector.weight']),
        "bias": th_to_jax(th_state['s_projector.bias']),
    }
    flax_params['t_embedder'] = convert_t_embedder(th_state)
    flax_params['y_embedder'] = convert_y_embedder(th_state)


    flax_params['enc_feat_rope'] = {
        'freqs_cos': th_to_jax(th_state['enc_feat_rope.freqs_cos']),
        'freqs_sin': th_to_jax(th_state['enc_feat_rope.freqs_sin']),
    }
    flax_params['dec_feat_rope'] = {
        'freqs_cos': th_to_jax(th_state['dec_feat_rope.freqs_cos']),
        'freqs_sin': th_to_jax(th_state['dec_feat_rope.freqs_sin']),
    }

    flax_params['enc_blocks'] = {}
    flax_params['dec_blocks'] = {}
    
    for i in range(encoder_depth):
        th_block = {k: v for k, v in th_state.items() if k.startswith(f"blocks.{i}.")}
        flax_params['enc_blocks'][i] = convert_block(th_block, i)

    for i in range(encoder_depth, depth):
        th_block = {k: v for k, v in th_state.items() if k.startswith(f"blocks.{i}.")}
        flax_params['dec_blocks'][i - encoder_depth] = convert_block(th_block, i)

    flax_params['final_layer'] = convert_final_layer(th_state)
    return flax_params

# === Example usage ===
# Assuming you have loaded a Flax checkpoint (e.g. as a nested dict called `flax_state`)
# and you have a PyTorch model (e.g. `torch_model`), you could convert and load the weights like:

# torch_state = convert_flax_to_torch(flax_state)
# torch_model.load_state_dict(torch_state)

# Note: In practice you may need to do additional debugging, adjust for mismatched dimensions,
# and verify that every parameter is converted correctly.
