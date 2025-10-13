"""File containing unittests for ddt modules."""

# built-in libs
import unittest

# external libs
from etils import epath
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import orbax.checkpoint as ocp

# deps
from tests.network_tests.ddt.ddt_torch import DiTwDDTHead
from networks.transformers import lightning_ddt_nnx
from networks.transformers import dit_nnx, port_torch_to_nnx as port


if __name__ == "__main__":
    
    torch.set_grad_enabled(False)

    th_model = DiTwDDTHead(
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
    )

    state_dict = torch.load("/home/nm3607/stage2_model.pt")
    th_model.load_state_dict(state_dict)

    model = lightning_ddt_nnx.LightningDDT(
        input_size=16,
        in_channels=768,
        patch_size=1,
        freq_embed_size=512,
        encoder_hidden_size=1152,
        encoder_num_heads=16,
        num_encoder_blocks=28,
        decoder_hidden_size=2048,
        decoder_num_heads=16,
        num_decoder_blocks=2,
        continuous_time_embed=True,
        qk_norm=False,
        use_rope=True,
        swiglu=True,
        adaln_shift=True,
        rms_norm=True,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 16, 768))
    y = jax.random.randint(jax.random.PRNGKey(1), (4,), 0, 1000)
    t = jax.random.uniform(jax.random.PRNGKey(2), (4,))

    th_x = torch.tensor(np.asarray(x), dtype=torch.float32).permute(0, 3, 1, 2)
    th_y = torch.tensor(np.asarray(y), dtype=torch.long)
    th_t = torch.tensor(np.asarray(t), dtype=torch.float32)

    graph, nnx_state = nnx.split(model)

    flax_state = flax.traverse_util.flatten_dict(nnx_state.to_pure_dict())

    for name, param in flax_state.items():
        print(name, param.shape)

    port_ckpt = port.convert_torch_to_flax(
        state_dict,
        depth=30,
        encoder_depth=28,
    )

    flat_port_state = flax.traverse_util.flatten_dict(port_ckpt)

    for name, param in flax_state.items():
        if name not in flat_port_state:
            print(f"Missing key: {name}")
            flat_port_state[name] = param
            continue
        port_param = flat_port_state[name]
        if param.shape != port_param.shape:
            print(f"Shape mismatch for {name}: {param.shape} vs {port_param.shape}")
            continue
    
    unflatten_ckpt = flax.traverse_util.unflatten_dict(flat_port_state)
    model = nnx.merge(graph, unflatten_ckpt)

    th_out = th_model(th_x, th_t, th_y)
    th_out = th_out.permute(0, 2, 3, 1).contiguous().numpy()

    jax_out = model(x, t, y)[0]

    print("diff: ", np.abs(th_out - jax_out).max())