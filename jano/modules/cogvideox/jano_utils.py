import torch


def get_formatted_latent(latents: torch.Tensor) -> torch.Tensor:
    '''
    input: (b, t, c, h, w)
    output: (c, t, h, w)
    '''
    a = latents.squeeze(0)
    return a.transpose(0,1)
    