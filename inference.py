import torch
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from build_vocabulary import build_vocab


class CaptionGenerator:
    def __init__(self, encoder_path, decoder_path, embed_size, hidden_size, vocab_size, num_layers, device, vocab):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
        self.vocab = vocab
        self.encoder, self.decoder = self.load_model()

    def generate_caption(self, image_path):
        processed_image = self.prepare_image_for_caption_generation(image_path).to(self.device)
        with torch.no_grad():
            features = self.encoder(processed_image)
            generated_tokens = self.decoder.sample(features)
        output = self.decode_caption(generated_tokens)
        return output

    def load_model(self):
        encoder = EncoderCNN(self.embed_size)
        decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)

        encoder.load_state_dict(torch.load(self.encoder_path))
        decoder.load_state_dict(torch.load(self.decoder_path))

        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        return encoder, decoder

    @staticmethod
    def prepare_image_for_caption_generation(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        image = Image.open(image_path).convert('RGB')
        transformed_image = transform(image).unsqueeze(0)
        return transformed_image

    @staticmethod
    def get_captions(captions_path):
        captions = {}
        with open(captions_path, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    image_id, caption = line.strip().split(',')[0], line.strip().split(',')[1]
                    if image_id in captions.keys():
                        captions[image_id].append(caption)
                    else:
                        captions[image_id] = [caption]
        return captions

    def decode_caption(self, generated_tokens):
        caption = []
        for token in generated_tokens[0]:  # Assuming batch size is 1
            word = self.vocab.idx_to_word(token.item())
            if word == '<end>':
                break
            if word != '<start>':
                caption.append(word)
        return ' '.join(caption)


if __name__ == '__main__':
    embed_size = 256
    hidden_size = 512
    vocab_size = 8876  # my vocab size
    num_layers = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions_path = 'data/captions.txt'
    captions = CaptionGenerator.get_captions(captions_path)
    vocab = build_vocab(captions)

    image_caption_generator = CaptionGenerator(
        encoder_path='encoder.ckpt',
        decoder_path='decoder.ckpt',
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device=device,
        vocab=vocab
    )

    generated_caption = image_caption_generator.generate_caption(image_path='test_image5.jpg')
    print(generated_caption)
