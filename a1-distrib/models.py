# models.py
import string
from sentiment_data import *
from utils import *
from nltk.corpus import stopwords
import math
import numpy as np
import random
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # raise Exception("Must be implemented")

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # sets to speed up computation
        stopwords_list = set(stopwords.words('english'))
        punc_list = set(string.punctuation)
        sentence_words = set()

        # removing stopwords, punctuation, whitespaces, and duplicates
        for unedited_word in sentence:
            unedited_word = unedited_word.lower()
            # removing punctuation within a word
            word = unedited_word.lower()
            if word not in stopwords_list and word != " " and word not in punc_list:
                sentence_words.add(word)

        # create counter for sparse storage
        feature_vector = Counter(sentence_words)

        # if you're adding to the initial training bag of words vector
        if add_to_indexer:
            # increment through feature vector and assign value (0) to keys (words)
            for word in feature_vector:
                # 0 denotes presence bc you're adding all words, no frequency calcs
                feature_vector[word] = 0
        # otherwise, don't change any feature vector

        return feature_vector

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # raise Exception("Must be implemented")

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # sets to speed up computation
        stopwords_list = set(stopwords.words('english'))
        punc_list = set(string.punctuation)
        bigram_sentence = []
        bigram_sentence_words = set()

        # removing stopwords, punctuation, whitespaces
        for unedited_word in sentence:
            unedited_word = unedited_word.lower()
            # removing punctuation within a word
            word = unedited_word.lower()
            if word not in stopwords_list and word != " " and word not in punc_list:
                bigram_sentence.append(word)

        # iterating through the sentence in pairs
        for pair_number in range(len(bigram_sentence) // 2):
            # indices for the two words in the pair
            first_word_index = 2 * pair_number
            second_word_index = 2 * pair_number + 1
            # check if you have the indicies for two pairs
            if second_word_index < len(bigram_sentence):
                word_pair = "&".join([bigram_sentence[first_word_index], bigram_sentence[second_word_index]])
            else:
                # otherwise you're on the last single word
                word_pair = bigram_sentence[first_word_index]
            # add combined word to the set
            bigram_sentence_words.add(word_pair)

        # create counter for sparse storage
        feature_vector = Counter(bigram_sentence_words)

        # if you're adding to the initial training bag of words vector
        if add_to_indexer:
            # increment through feature vector and assign value (0) to keys (words)
            for word in feature_vector:
                # 0 denotes presence bc you're adding all words, no frequency calcs
                feature_vector[word] = 0
        # otherwise, don't change any feature vector

        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # raise Exception("Must be implemented")

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # Sets to speed up computation
        stopwords_list = set(stopwords.words('english'))
        punc_list = set(string.punctuation)
        sentence_words = []

        # Removing stopwords, punctuation, whitespaces, and duplicates
        for unedited_word in sentence:
            unedited_word = unedited_word.lower()
            # Removing punctuation within a word
            word = unedited_word.lower()
            if word not in stopwords_list and word != " " and word not in punc_list:
                sentence_words.append(word)

        # Create a counter for sparse storage
        feature_vector = Counter(sentence_words)

        # If you're adding to the initial training bag of words vector
        if add_to_indexer:
            # Increment through the feature vector and assign the actual term frequency to keys (words)
            for word in sentence_words:
                feature_vector[word] = sentence_words.count(word)

        return feature_vector

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor, bag_of_words_vector, weight_vector, indexer: Indexer):
        self.feature_extractor = feature_extractor
        self.sentence_vector = Counter()
        self.bag_of_words_vector = bag_of_words_vector
        self.weight_vector = weight_vector
        self.indexer = indexer

        # raise Exception("Must be implemented")

    def predict(self, sentence: List[str]) -> int:
        # takes sentence and makes prediction

        # extract features of sentence, not adding to bag of word indexer, bool=false by default
        sentence_features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)

        # create lists of counters to index
        bag_of_words_vector_list = list(self.bag_of_words_vector.keys())
        weight_vector_list = list(self.weight_vector.items())

        # create a list to store nonzero indicies in sparse sentence Counter
        nonzero_index = []
        # iterate through bag of words and mark which extracted sentence features are in the bag of words vector
        for word in self.bag_of_words_vector:
            # if the word appears at all, use 1/0 encoding to create a bag of words length vector
            if word in sentence_features:
                self.bag_of_words_vector[word] = 1
                # add index to list of nonzero items list as a marker of what to multiply
                nonzero_index.append(bag_of_words_vector_list.index(word))
            else:
                self.bag_of_words_vector[word] = 0
        # rename edited vector
        self.sentence_vector = self.bag_of_words_vector.copy()
        sentence_vector_list = list(self.bag_of_words_vector.items())

        # dot product between the feature vector and weight vector
        dot_prod = 0.
        for idx in nonzero_index:
            relevant_feat = sentence_vector_list[idx][1]
            relevant_weight = weight_vector_list[idx][1]
            dot_prod += relevant_feat * relevant_weight

        # prediction when 1=positive and 0=negative
        return 1 if dot_prod > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor, bag_of_words_vector, weight_vector, indexer: Indexer):
        self.feature_extractor = feature_extractor
        self.sentence_vector = Counter()
        self.bag_of_words_vector = bag_of_words_vector
        self.weight_vector = weight_vector
        self.indexer = indexer
        self.sigmoid = float()

        # raise Exception("Must be implemented")

    def predict(self, sentence: List[str]) -> int:
        # takes sentence and makes prediction

        # extract features of sentence, not adding to bag of word indexer, bool=false by default
        sentence_features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)

        # create lists of counters to index
        bag_of_words_vector_list = list(self.bag_of_words_vector.keys())
        weight_vector_list = list(self.weight_vector.items())

        # create a list to store nonzero indicies in sparse sentence Counter
        nonzero_index = []
        # iterate through bag of words and mark which extracted sentence features are in the bag of words vector
        for word in self.bag_of_words_vector:
            # if the word appears at all, use 1/0 encoding to create a bag of words length vector
            if word in sentence_features:
                self.bag_of_words_vector[word] = 1
                # add index to list of nonzero items list as a marker of what to multiply
                nonzero_index.append(bag_of_words_vector_list.index(word))
            else:
                self.bag_of_words_vector[word] = 0
        # rename edited vector
        self.sentence_vector = self.bag_of_words_vector.copy()
        sentence_vector_list = list(self.bag_of_words_vector.items())

        # dot product between the feature vector and weight vector
        dot_prod = 0.
        for idx in nonzero_index:
            relevant_feat = sentence_vector_list[idx][1]
            relevant_weight = weight_vector_list[idx][1]
            dot_prod += relevant_feat * relevant_weight

        self.sigmoid = 1/(1+np.exp(-dot_prod))
        # print("DP:", dot_prod)
        # print("PROB", self.sigmoid)

        # prediction when 1=positive and 0=negative
        return 1 if self.sigmoid >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, bag_of_words_vector=None,
                     weight_vector=None) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # raise Exception("Must be implemented")

    # input: trained set, feature extractor
    # output: returns classifier

    # feature indexer
    indexer = feat_extractor.get_indexer()
    perceptron_classifier = PerceptronClassifier(feat_extractor, indexer, bag_of_words_vector, weight_vector)
    bag_of_words_vector = Counter()

    # iterate through training exs to get weight vector ahead of time
    for example in train_exs:
        training_features = feat_extractor.extract_features(example.words, add_to_indexer=True)
        # update overall bag_of_words_vector to maintain binary encoding vs frequency
        bag_of_words_vector.update({word: 0 for word in training_features})
    # add the final bag_of_words_vector as an argument of the perceptron classifier to be used in predict class
    perceptron_classifier.bag_of_words_vector = bag_of_words_vector

    # initially set the weight vector to be the bag of words vector w/ all 0s
    weight_vector = bag_of_words_vector.copy()
    # add the weight_vector as an argument of the perceptron classifier to be used in predict class
    # starts off w/ all 0s --> updates at the end of the 3 epochs
    perceptron_classifier.weight_vector = weight_vector

    # epochs = # different orders to train model
    for epoch in range(4):
        # randomly shuffle training exs, set random seed for each epoch for testing
        random.seed(7*epoch)
        random.shuffle(train_exs)

        # set step size
        alpha = 0.5 - (epoch * 0.1)

        # iterate through training exs to train model
        for example in range(len(train_exs)):
            # extract sentiment label
            true_sentiment = train_exs[example].label
            # extract sentence
            sentence = train_exs[example].words
            # predict sentiment label using perceptron classifier
            pred_sentiment = perceptron_classifier.predict(sentence)
            # sentence_vector from training example
            sentence_vector = perceptron_classifier.sentence_vector
            # if wrong prediction: pred sent=0, true sent=1 --> weight vector=weight vector+(step size*sentence vector)
            if pred_sentiment < true_sentiment:
                for word in weight_vector:
                    # skip over 0 values to speed up
                    if sentence_vector[word] != 0:
                        weight_vector[word] = round(weight_vector[word] + (alpha * sentence_vector[word]), 4)
            # if wrong prediction: pred sent=1, true sent=0 --> weight vector=weight vector-(step size*sentence vector)
            if pred_sentiment > true_sentiment:
                for word in weight_vector:
                    # skip over 0 values to speed up
                    if sentence_vector[word] != 0:
                        weight_vector[word] = round(weight_vector[word] - (alpha * sentence_vector[word]), 4)

    return perceptron_classifier

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, bag_of_words_vector=None,
                     weight_vector=None) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # raise Exception("Must be implemented")
    # input: trained set, feature extractor
    # output: returns classifier

    # feature indexer
    indexer = feat_extractor.get_indexer()
    logreg_classifier = LogisticRegressionClassifier(feat_extractor, indexer, bag_of_words_vector, weight_vector)
    bag_of_words_vector = Counter()

    # iterate through training exs to get weight vector ahead of time
    for example in train_exs:
        training_features = feat_extractor.extract_features(example.words, add_to_indexer=True)
        # update overall bag_of_words_vector to maintain binary encoding vs frequency
        bag_of_words_vector.update({word: 0 for word in training_features})
    # add the final bag_of_words_vector as an argument of the perceptron classifier to be used in predict class
    logreg_classifier.bag_of_words_vector = bag_of_words_vector

    # initially set the weight vector to be the bag of words vector w/ all 0s
    weight_vector = bag_of_words_vector.copy()
    # add the weight_vector as an argument of the perceptron classifier to be used in predict class
    # starts off w/ all 0s --> updates at the end of the 3 epochs
    logreg_classifier.weight_vector = weight_vector

    # epochs = # different orders to train model
    for epoch in range(4):
        # randomly shuffle training exs, set random seed for each epoch for testing
        random.seed(7*epoch)
        random.shuffle(train_exs)

        # set step size
        alpha = 0.5 - (epoch * 0.1)

        # iterate through training exs to train model
        for example in range(len(train_exs)):
            # extract sentiment label
            true_sentiment = train_exs[example].label
            # extract sentence
            sentence = train_exs[example].words
            # predict sentiment label using perceptron classifier
            pred_sentiment = logreg_classifier.predict(sentence)
            # sentence_vector from training example
            sentence_vector = logreg_classifier.sentence_vector
            sigmoid = logreg_classifier.sigmoid
            # if wrong prediction:
            if pred_sentiment != true_sentiment:
                for word in weight_vector:
                    # skip over 0 values to speed up
                    if sentence_vector[word] != 0:
                        weight_vector[word] = round(weight_vector[word] - (alpha * sigmoid * sentence_vector[word]), 4)

    return logreg_classifier

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model