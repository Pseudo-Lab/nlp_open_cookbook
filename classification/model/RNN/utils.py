from konlpy.tag import Kkma, Okt, Komoran
from torchtext.vocab import build_vocab_from_iterator

def get_tokenizer(name):
    """get tokenizer

    Args:
        name (str): name of tokenizer(kkma, okt, komoran)

    Returns:
        tokenizer
    """
        if name == 'kkma':
            tokenizer = Kkma()
        
        elif name == 'okt':
            tokenizer = Okt()
        
        elif name == 'komoran':
            tokenizer = Komoran()
        
        return tokenizer

def yield_tokens(sequences, tokenizer):
    """generate token

    Args:
        sequences (list): list of sequences
        tokenizer (tokenizer)

    Yields:
        list: tokens of sentence
    """
    for sentence in sequences:
        yield tokenizer.nouns(sentence)

def set_vocab(sequences, tokenizer, vocab_size):
    """set vocabulary dictionary

    Args:
        sequences (list): list of sequences
        tokenizer (tokenizer)
        vocab_size (int): max length of vocabulary dictionary

    Returns:
        vocab: vocab object
    """
    
    vocab = build_vocab_from_iterator(yield_tokens(sequences, tokenizer), 
                                      specials=['<unk>'], 
                                      max_tokens=vocab_size)
    
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def integer_encoding(sequences, tokenizer, vocab):
    """encoding sequences into integer vectors

    Args:
        sequences (list): list of sequences
        tokenizer (tokenizer)
        vocab (vocab): vocab object

    Returns:
        list: list of integer vectors
    """
    sequences = [vocab(tokenizer.nouns(sentence)) for sentence in sequences]
    return sequences