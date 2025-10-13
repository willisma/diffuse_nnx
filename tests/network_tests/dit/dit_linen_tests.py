import unittest
import sys

# add current directory to path
sys.path.append('.')

import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from networks.transformers.dit import DiT as DiT_jax
from networks.transformers.utils import get_2d_sincos_pos_embed, to_2tuple
from .dit_torch import DiT as DiT

key = random.PRNGKey(0)

def convert_jax_to_torch(nested_jax_params, num_blocks, hidden_size, patch_size):
    """
    Convert JAX parameters to PyTorch parameters.
    """
    torch_params = {}

    # process input embedding
    conv_kernel = nested_jax_params["PatchEmbed_0"]["Conv_0"]["kernel"]
    conv_bias = nested_jax_params["PatchEmbed_0"]["Conv_0"]["bias"]
    torch_params["x_embedder.proj.weight"] = torch.from_numpy(np.transpose(np.array(conv_kernel), (3, 2, 0, 1)))
    torch_params["x_embedder.proj.bias"] = torch.from_numpy(np.array(conv_bias))

    # process time embedding
    t_mlp_0_weight = nested_jax_params["TimestepEmbedder_0"]["Dense_0"]["kernel"]
    t_mlp_0_bias = nested_jax_params["TimestepEmbedder_0"]["Dense_0"]["bias"]
    t_mlp_1_weight = nested_jax_params["TimestepEmbedder_0"]["Dense_1"]["kernel"]
    t_mlp_1_bias = nested_jax_params["TimestepEmbedder_0"]["Dense_1"]["bias"]
    torch_params["t_embedder.mlp.0.weight"] = torch.from_numpy(np.array(t_mlp_0_weight.T))
    torch_params["t_embedder.mlp.0.bias"] = torch.from_numpy(np.array(t_mlp_0_bias))
    torch_params["t_embedder.mlp.2.weight"] = torch.from_numpy(np.array(t_mlp_1_weight.T))
    torch_params["t_embedder.mlp.2.bias"] = torch.from_numpy(np.array(t_mlp_1_bias))

    # process label embedding
    label_embedding = nested_jax_params["LabelEmbedder_0"]["Embed_0"]["embedding"]
    torch_params["y_embedder.embedding_table.weight"] = torch.from_numpy(np.array(label_embedding))

    # process blocks
    for block_idx in range(num_blocks): 
        jax_prefix = f"DiTBlock_{block_idx}"
        torch_prefix = f"blocks.{block_idx}."

        # process attention
        q = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["query"]["kernel"].reshape(hidden_size, -1)
        k = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["key"]["kernel"].reshape(hidden_size, -1)
        v = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["value"]["kernel"].reshape(hidden_size, -1)
        qkv_weight = np.concatenate([q, k, v], axis=1).T
        torch_params[f"{torch_prefix}attn.qkv.weight"] = torch.from_numpy(np.array(qkv_weight))

        # process attention bias
        q_b = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["query"]["bias"].reshape(-1)
        k_b = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["key"]["bias"].reshape(-1)
        v_b = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["value"]["bias"].reshape(-1)
        qkv_bias = np.concatenate([q_b, k_b, v_b])
        torch_params[f"{torch_prefix}attn.qkv.bias"] = torch.from_numpy(np.array(qkv_bias))

        # process attention out projection
        out_weight = nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["out"]["kernel"].reshape(-1, hidden_size).T
        torch_params[f"{torch_prefix}attn.proj.weight"] = torch.from_numpy(np.array(out_weight))
        torch_params[f"{torch_prefix}attn.proj.bias"] = torch.from_numpy(np.array(
            nested_jax_params[jax_prefix]["MultiHeadDotProductAttention_0"]["out"]["bias"]
        ))

        # process mlp
        mlp_fc1_weight = nested_jax_params[jax_prefix]["Mlp_0"]["Dense_0"]["kernel"]
        mlp_fc1_bias = nested_jax_params[jax_prefix]["Mlp_0"]["Dense_0"]["bias"]
        mlp_fc2_weight = nested_jax_params[jax_prefix]["Mlp_0"]["Dense_1"]["kernel"]
        mlp_fc2_bias = nested_jax_params[jax_prefix]["Mlp_0"]["Dense_1"]["bias"]
        torch_params[f"{torch_prefix}mlp.fc1.weight"] = torch.from_numpy(np.array(mlp_fc1_weight.T))
        torch_params[f"{torch_prefix}mlp.fc1.bias"] = torch.from_numpy(np.array(mlp_fc1_bias))
        torch_params[f"{torch_prefix}mlp.fc2.weight"] = torch.from_numpy(np.array(mlp_fc2_weight.T))
        torch_params[f"{torch_prefix}mlp.fc2.bias"] = torch.from_numpy(np.array(mlp_fc2_bias))

        # process adaLN
        torch_params[f"{torch_prefix}adaLN_modulation.1.weight"] = torch.from_numpy(np.array(
            nested_jax_params[jax_prefix]["Dense_0"]["kernel"].T
        ))
        torch_params[f"{torch_prefix}adaLN_modulation.1.bias"] = torch.from_numpy(np.array(
            nested_jax_params[jax_prefix]["Dense_0"]["bias"]
        ))

    # process final layer
    final_linear_weight = nested_jax_params["FinalLayer_0"]["Dense_1"]["kernel"]
    final_linear_bias = nested_jax_params["FinalLayer_0"]["Dense_1"]["bias"]
    torch_params["final_layer.linear.weight"] = torch.from_numpy(np.array(final_linear_weight.T))
    torch_params["final_layer.linear.bias"] = torch.from_numpy(np.array(final_linear_bias))
    torch_params["final_layer.adaLN_modulation.1.weight"] = torch.from_numpy(np.array(
        nested_jax_params["FinalLayer_0"]["Dense_0"]["kernel"].T
    ))
    torch_params["final_layer.adaLN_modulation.1.bias"] = torch.from_numpy(np.array(
        nested_jax_params["FinalLayer_0"]["Dense_0"]["bias"]
    ))

    torch_params["pos_embed"] =  torch.from_numpy(np.array(get_2d_sincos_pos_embed(hidden_size, to_2tuple(patch_size)))[None, ...])

    return torch_params

class TestDiT(unittest.TestCase):

    def test_dit(self):
        x = random.normal(key, (2, 256, 256, 3))
        c = random.randint(key, (2,), 0, 1000)
        t = random.randint(key, (2,), 0, 1000)

        # enable jax 64bit
        jax.config.update('jax_enable_x64', True)

        x = jax.device_put(x, jax.devices('cpu')[0])
        c = jax.device_put(c, jax.devices('cpu')[0])
        t = jax.device_put(t, jax.devices('cpu')[0])

        with jax.default_device(jax.devices("cpu")[0]):
            model = DiT_jax(
                input_size=256, patch_size=16, in_channels=3, num_classes=1000, hidden_size=768, num_heads=4, depth=12
            )
            params = model.init(key, x, t, c, training=False)['params']
            jit_apply = jit(model.apply, static_argnames=('training',))
            result = jit_apply({'params': params}, x, t, c, training=False).transpose(0, 3, 1, 2)

            model_torch = DiT(
                input_size=256, patch_size=16, in_channels=3, num_classes=1000, hidden_size=768, num_heads=4, depth=12
            )
            model_torch.load_state_dict(convert_jax_to_torch(params, num_blocks=12, hidden_size=768, patch_size=16))

            x = torch.tensor(np.array(x)).permute(0, 3, 1, 2)
            t = torch.tensor(np.array(t))
            c = torch.tensor(np.array(c))

            result_torch = model_torch(x, t, c)
            self.assertTrue(jnp.allclose(result, result_torch.detach().numpy(), atol=1e-4))

if __name__ == '__main__':
    unittest.main()
