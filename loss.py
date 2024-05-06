import torch
import torch.nn as nn

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # print(logits, torch.arange(len(logits)))
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # print('caption_loss')
    caption_loss = contrastive_loss(similarity)
    # print('image_loss')
    image_loss = contrastive_loss(similarity.t())
    # print('caption_loss, image_loss')
    # print(caption_loss, image_loss)
    return (caption_loss + image_loss) / 2.0