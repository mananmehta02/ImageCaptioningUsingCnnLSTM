
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Adding special tokens
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        """adds a word to the vocabulary"""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        """
        returns the index of the word in the vocabulary
        """
        return len(self.word2idx)

    def idx_to_word(self, idx):
        return self.idx2word[idx]

def build_vocab(captions):
    vocab = Vocabulary()
    for caption in captions:
        tokens = caption.split(' ')
        for token in tokens:
            vocab.add_word(token)
    return vocab