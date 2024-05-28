import os

import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image


class Flickr8k(Dataset):
    def __init__(self, image_path, caption_path, transform=None):
        self.image_path = image_path
        self.image_labels = os.listdir(self.image_path)
        self.caption_path = caption_path
        self.captions = self.get_captions()
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_labels[idx])
        image = Image.open(img_path)
        captions = self.captions[self.image_labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, captions

    def get_captions(self):
        captions = {}
        with open(self.caption_path, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    image_id, caption = line.strip().split(',')[0], line.strip().split(',')[1]
                    if image_id in captions.keys():
                        captions[image_id].append(caption)
                    else:
                        captions[image_id] = [caption]
        return captions


def custom_collate_fn(batch):
    images, all_captions = zip(*batch)

    # Stack images into a batch
    images = torch.stack(images, dim=0)

    # Flatten the list of captions
    captions = []
    lengths = []
    for caption_list in all_captions:
        lengths.append(len(caption_list))
        captions.extend(caption_list)

    return images, captions, lengths


def get_loader(image_path, caption_path, transform=None, batch_size=8, shuffle=True, num_workers=1):
    flickr = Flickr8k(image_path=image_path, caption_path=caption_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=flickr, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, collate_fn=custom_collate_fn)
    print("Executed get loader successfully")
    return data_loader
