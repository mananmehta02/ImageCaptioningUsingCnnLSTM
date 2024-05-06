import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
# from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(num_epochs=5, learning_rate=0.001, model_path='.'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image_path = ""
    caption_path = ""
    vocab = []
    data_loader = get_loader(image_path=image_path, caption_path=caption_path, transform=transform, shuffle=True)

    encoder = EncoderCNN(embed_size=256).to(device)
    decoder = DecoderRNN(embed_size=256, vocab_size=len(vocab), hidden_size=512, num_layers=1, max_seq_length=20)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder.ckpt'))

    torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder.ckpt'))


if __name__ == '__main__':
    train(model_path='.')