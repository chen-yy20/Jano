from .attention_processor import JointAttnProcessor_jano


def transformer_to_std_shape(latents):
    return latents.unsqueeze(2).squeeze(0)    


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def wrap_sd3_model_with_jano(model):
    for i, block in enumerate(model.transformer_blocks):
        block.attn.processor = JointAttnProcessor_jano(layer_id=i)