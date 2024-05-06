import torch
from config import CLIPTextConfig, CLIPVisionConfig, CLIPConfig
from model.clip_model import CLIPModel


text_config = CLIPTextConfig()
vision_config = CLIPVisionConfig()
clip_config = CLIPConfig(text_config, vision_config)
model = CLIPModel(clip_config)

batch = torch.load('data_test/batch.pt')

outputs = model(
    **batch,
    return_loss=True
)

print(outputs)