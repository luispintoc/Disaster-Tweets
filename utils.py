from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import TweetTokenizer
import os, re, string, sys
import numpy as np
import tokenization

mispell_dict = {"aren't" : "are not",
"arent" : "are not",
"can't" : "can not",
"cant" : "can not",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"didnt" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"dont" : "do not",
"hadn't" : "had not",
"hadnt" : "had not",
"hasn't" : "has not",
"hasnt" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"id" : "I would",
"i'll" : "I will",
"i'm" : "I am",
"im" : "i am",
"isn't" : "is not",
"isnt" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"ive" : "i have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"shed" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"theyd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"werent" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"whats" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"whod" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wont" : "will not",
"wouldn't" : "would not",
"wouldnt" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"youve" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"u": "you",
"cannot": "can not"}

stop = set(stopwords.words('english'))
stop = ENGLISH_STOP_WORDS.union(stop)


# Cleaning of tweets
def process_tweet(tweet):
	tweet = tweet.strip(' ')
	tweet = re.sub(r'\$\w*', '', tweet)
	tweet = re.sub(r'^RT[\s]+', '', tweet)
	tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	tweet = re.sub(r'#', '', tweet)
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
	tweet_tokens = tokenizer.tokenize(tweet)

	tweet_clean = ''

	for word in tweet_tokens:
		if word in mispell_dict.keys():
			word = mispell_dict[word].lower()

		if (word not in string.punctuation and word not in stop):
			tweet_clean+= (' '+ word)

	return tweet_clean



# Tokenizer
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)