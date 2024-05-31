import os

import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image
from build_vocabulary import build_vocab


class Flickr8k(Dataset):
    def __init__(self, image_path, caption_path, transform=None):
        self.image_path = image_path
        self.image_labels = os.listdir(self.image_path)
        self.caption_path = caption_path
        self.captions = self.get_captions()
        self.transform = transform
        self.vocab = build_vocab(self.captions)
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_labels[idx])
        image = Image.open(img_path).convert("RGB")
        captions = self.captions[self.image_labels[idx]]
        tokenized_captions = []
        for caption in captions:
            tokenized_captions.append(self.tokenize(caption))
        if self.transform:
            image = self.transform(image)
        return image, tokenized_captions

    def tokenize(self, caption):
        tokens = caption.strip().lower().split()
        tokenized_caption = [self.vocab.word2idx['<start>']]
        for token in tokens:
            tokenized_caption.append(self.vocab.word2idx[token] if token in self.vocab.word2idx else self.vocab.word2idx['<unk>'])
        tokenized_caption.append(self.vocab.word2idx['<end>'])
        return tokenized_caption

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
    images, captions_list = zip(*batch)

    # Stack images into a batch
    images = torch.stack(images, dim=0)

    padded_captions = []
    lengths = []
    for captions in captions_list:
        lengths.extend([len(caption) for caption in captions])
        padded_captions.extend([torch.tensor(caption) for caption in captions])

    padded_captions = pad_sequence(padded_captions, batch_first=True, padding_value=0)
    return images, padded_captions, lengths


def get_loader(image_path, caption_path, transform=None, batch_size=8, shuffle=True, num_workers=1):
    flickr = Flickr8k(image_path=image_path, caption_path=caption_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=flickr, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, collate_fn=custom_collate_fn)
    print("Executed get loader successfully")
    return data_loader, flickr.vocab_size
