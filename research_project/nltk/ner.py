import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, ClassifierBasedTagger
from nltk.stem.snowball import SnowballStemmer
from nltk import TaggerI, ChunkParserI
from collections.abc import Iterable
from nltk.chunk.util import conlltags2tree

class ConsecutiveNEChunkTagger(TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                feature_set = nechunk_features(untagged_sent, i, history)
                train_set.append( (feature_set, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            feature_set = nechunk_features(sentence, i, history)
            tag = self.classifier.classify(feature_set)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNEChunker(ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [ [((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = ConsecutiveNEChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

def nechunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {'pos': pos}

def read_data(path):
    tweets = []
    with open(path) as train_file:
        iob_pos_tweet = []
        prev_line = ''
        prev_tokens = []
        counter = 0
        for line in train_file:            
            if line.isspace():
                if not iob_pos_tweet:
                    #print(prev_line)
                    #print(counter)
                    print(prev_tokens)
                    continue
                    raise ValueError('tweet empty')
                tweets.append(iob_pos_tweet)
                iob_pos_tweet = []
            else:
                line = line.strip()
                tokens = line.split()
                if not tokens:
                    raise ValueError('tokens empty')
                iob_pos_tweet.append( ((tokens[0], tokens[1]), tokens[2]) )
                prev_tokens = tokens
            prev_line = line
            counter = counter + 1
    return tweets

#research_project/nltk/
training_data = read_data('train.txt')
chunker = NamedEntityChunker(training_data)
test_data = read_data('test.txt')
score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_data])
print('precision: ', score.precision())
print('recall: ', score.recall())
print('f1: ', score.f_measure())
#tweets = [pos_tag(t) for t in tweets]
#sent = nltk.corpus.treebank.tagged_sents()[22]
#tweets = [nltk.ne_chunk(t) for t in tweets]