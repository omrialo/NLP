from ex3_preProcessing import *
from ex3_models import *
from LSTM import LSTM_train_best_model,LSTM_load_best_model, make_predictions
from config import tfidfParams
import csv
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def load_best_model():
    """ This function loads the best model which is LSTM network
    returns:
        best_model: the LSTM trained model
    """
    best_model = LSTM_load_best_model()
    return best_model


def train_best_model():
    """ This function trains from scratch the best model which is LSTM network
        Args:
            None, but this function assumes 'trump_train.tsv' is located in the current directory
        returns:
            best_model: the LSTM trained model
    """
    best_model = LSTM_train_best_model()
    return best_model

def predict(m, fn):
    """ This generates predictions given a model and test data
        if the model is not LSTM, it will handle it as well (altough we assumed it will be used only with LSTM)
        Args:
            m(model): fully trained model
            fn(str): full path to test file
        returns:
            predictions(list): list of predictions made by the model
    """
    try:
        predictions = make_predictions(m, fn, version='title', threshold=0.5)
    except:
        try:
            train_data = pd.read_csv('trump_train.tsv', sep='\t',names=['<tweet id>', '<user handle>', '<tweet text>', '<time stamp>',
                                            '<device> = Label'], engine='python', quoting=csv.QUOTE_NONE)
            test_data = pd.read_csv(fn, sep='\t', names=['<user handle>', '<tweet text>', '<time stamp>'],
                                    engine='python', quoting=csv.QUOTE_NONE)
            train_data, test_data = extractTimeBaseFeatures(train_data, test_data)
            train_data, test_data = extractTrumpUniqueFeatures(train_data, test_data)
            train_data = createLabel(train_data)
            train_data, test_data = getTfidfFFeatures(train_data, test_data, tfidfParams)
            predictions = m.predict(test_data)
        except:
            train_data = pd.read_csv('trump_train.tsv', sep='\t',names=['<tweet id>', '<user handle>', '<tweet text>', '<time stamp>',
                                            '<device> = Label'], engine='python', quoting=csv.QUOTE_NONE)
            test_data = pd.read_csv(fn, sep='\t', names=['<user handle>', '<tweet text>', '<time stamp>'],
                                    engine='python', quoting=csv.QUOTE_NONE)
            train_data = train_data.dropna()
            train_labels = train_data['isStaffer']
            train_data = train_data.drop('isStaffer', 1)
            NN = NeuralNetwork(input_dim=train_data.shape[1], hidden_dim=[32, 16, 6], output_dim=2)
            NN.train_network(train_data, train_labels)
            pred = NN.model(torch.tensor(test_data.values).float())
            predictions = torch.argmax(pred.data, dim=1)
    return predictions

train_best_model()