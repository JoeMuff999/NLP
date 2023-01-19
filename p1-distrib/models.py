# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import nltk
import string
import matplotlib


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self) -> Indexer:
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
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
        print("Created UnigramFeatureExtractor")

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        return Counter(sentence)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, featurizer : FeatureExtractor):
        self.weights = weights
        self.featurzier = featurizer
        print("Created Logistic Regression Classifier")

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        features = self.featurzier.extract_features(ex_words, False)
        
        weighted_feature_sum = 0
        #calculate logistic regression value p(y | x; weights)
        for word in features.elements():
            index = self.featurzier.get_indexer().index_of(word)
            if index > -1:
                weighted_feature_sum += self.weights[index]
            
        exp_weights = np.exp(weighted_feature_sum)
        probability = exp_weights/(1 + exp_weights)
        # theres probably a cleaner way to do this...
        if probability > 0.5:
            return 1
        return 0

# training_schedule = {"name" : "constant_0-01", "learning_rate" : .01}
# training_schedule = {"name" : "constant_0-1", "learning_rate" : .1}
# training_schedule = {"name" : "constant_1", "learning_rate" : 1}
# training_schedule = {"name" : "per_step", "learning_rate" : 1}
training_schedule = {"name" : "per_epoch", "learning_rate" : 1}
EPOCHS = 15

#average log-likelihoods per epoch

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    log_likelihoods = np.zeros(EPOCHS*len(train_exs))
    ll_per_epoch = np.zeros(EPOCHS)
    dev_accuracy_per_epoch = np.zeros(EPOCHS)
    #fixed size numpy.ndarray for weights? for now, we will dynamically increase size
    weights = np.zeros(len(feat_extractor.get_indexer()))
    # train the model bro...
    dev_exs = read_sentiment_examples('data/dev.txt')

    random.seed(1)
    learning_rate = training_schedule["learning_rate"]
    step_count = 0
    for epoch in range(EPOCHS):
        random.shuffle(train_exs)
        for example in train_exs:
            x = example.words
            features = feat_extractor.extract_features(x, False)
            #calculate logistic regression value p(y | x; weights)
            weighted_feature_sum = 0
            for word in features.elements():
                index = feat_extractor.get_indexer().index_of(word)
                if index >= 0:
                    weighted_feature_sum += weights[index]

            exp_weights = np.exp(weighted_feature_sum)
            probability = exp_weights/(1 + exp_weights)
            update_val = example.label - probability

            for word in features.keys():
                weights[feat_extractor.get_indexer().index_of(word)] += learning_rate * update_val
            
            log_likelihood = np.log(probability) * example.label + np.log(1 - probability)*(1-example.label)
            log_likelihoods[step_count] = log_likelihood
            step_count += 1
            if training_schedule["name"] == "per_step":
                learning_rate = 1.0/step_count
        if training_schedule["name"] == "per_epoch":
            learning_rate = 1.0/(epoch+1)
        ll_per_epoch[epoch] = sum(log_likelihoods[epoch*len(train_exs):(epoch+1)*len(train_exs)])/len(train_exs)       

        # get dev accuracy
        temp_classifier = LogisticRegressionClassifier(weights, feat_extractor)
        for example in dev_exs:
            if temp_classifier.predict(example.words) == example.label:
                dev_accuracy_per_epoch[epoch] += 1/len(dev_exs)

    np.savetxt('mydata/training/' + str(training_schedule['name']) + '.csv', log_likelihoods)
    np.savetxt('mydata/training/' + str(training_schedule['name']) + '-epoch.csv', ll_per_epoch)
    np.savetxt('mydata/training/' + str(training_schedule['name']) + '-dev.csv', dev_accuracy_per_epoch)


    return LogisticRegressionClassifier(weights, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
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
        #remove stop words
        # nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
        #build vocabulary
        vocab = Indexer()
        for example in train_exs:
            for word in example.words:
                if word not in string.punctuation and word not in stopwords:
                    vocab.add_and_get_index(word)
        feat_extractor = UnigramFeatureExtractor(vocab)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
