import os
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, image_folder, annotations, tokenizer, transform=None):
        self.image_folder = image_folder
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        # Text
        text_inputs = self.tokenizer(
            self.annotations.iloc[idx]['comment'], 
            max_length=77,
            padding='max_length',
            return_tensors="pt"
        )

        # Image
        image = Image.open(os.path.join(self.image_folder, self.annotations.iloc[idx]['image_name']))
        if self.transform: image_inputs = self.transform(image)
        
        outputs = {
            'input_ids': text_inputs['input_ids'][0],
            'attention_mask': text_inputs['attention_mask'][0],
            'pixel_values': image_inputs
        }
        return outputs