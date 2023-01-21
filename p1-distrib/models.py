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


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    #fixed size numpy.ndarray for weights? for now, we will dynamically increase size
    weights = np.zeros(len(feat_extractor.get_indexer()))
    # train the model bro...
    random.seed(1)
    learning_rate = 0.1
    for _ in range(10):
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

    # print(weights)
    # print(feat_extractor.get_indexer().get_object(1))

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

import torch.nn as nn

class FFNN(nn.Module):

    def __init__(self, inp, hid, out, word_embeddings : WordEmbeddings):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        # self.dropout = nn.Dropout()
        self.word_embeddings = word_embeddings

    def forward(self, _input : List[str]):
        """

        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        encoded_input = np.zeros(self.word_embeddings.get_embedding_length())
        for word in _input:
            encoded_input = np.add(encoded_input, self.word_embeddings.get_embedding(word))
            
        encoded_input = encoded_input / len(_input)
        encoded_input = torch.tensor(encoded_input).float()
        return self.log_softmax(self.W(self.g(self.V(encoded_input))))

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network : FFNN, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings
        print("Created NeuralSentimentClassifier")

    def predict(self, ex_words: List[str]) -> int:
        self.network.eval() # dont change weights anymore
        return torch.argmax(self.network.forward(ex_words)).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # RUN TRAINING AND TEST
    num_output_classes = 2
    num_epochs = 15
    ffnn = FFNN(word_embeddings.get_embedding_length(), 10, num_output_classes, word_embeddings)
    ffnn.train()
    initial_learning_rate = 0.0015
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = train_exs[idx].words
            y = train_exs[idx].label
            y_onehot = torch.zeros(num_output_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            log_probs = ffnn.forward(x)
            # print(log_probs)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return NeuralSentimentClassifier(ffnn, word_embeddings)
