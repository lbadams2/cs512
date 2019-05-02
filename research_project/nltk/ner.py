import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

tweets = []
#../osu/tweets.txt
with open('research_project/nltk/test.txt') as tweet_file:
    tweet = []
    for line in tweet_file:        
        if line.isspace():
            tweets.append(tweet)
            tweet = []
        else:
            line = line.strip()
            word = line.split()[0]
            tweet.append(word)
    #tweets = [word_tokenize(tweet) for tweet in tweet_file]

tweets = [pos_tag(t) for t in tweets]
sent = nltk.corpus.treebank.tagged_sents()[22]
tweets = [nltk.ne_chunk(t) for t in tweets]

print('')

