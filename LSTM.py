import re
import pandas as pd
from torchtext.data import Field, LabelField, TabularDataset, Dataset, Example, BucketIterator


def normalize_text(text):
    """Returns a normalized string based on the specified string.
       Args:
           text (str): the text to normalize
       Returns:
           string. the normalized text.
    """
    nt = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()>])(?! )', r' ', text)
    return nt.lower()


from torchtext.vocab import GloVe


embedding = GloVe(name='twitter.27B', dim='25')
df = pd.read_csv('trump_train.tsv', sep='\\t',
                 names=['id_num', 'name', 'tweet', 'dt', 'label'], engine='python')

df = df.drop(df[(df.label != 'android') & (df.label != 'iphone')].index)
df['tweet'] = df['tweet'].apply(lambda x: normalize_text(x))
df['label'] = df['label'].apply(lambda x: 0 if x == 'android' else 1)
# df = add_date_features(df)
df = df.drop(df[(df.tweet == '')].index)

from sklearn.model_selection import train_test_split
destination_folder = '/content/my-drive/My Drive/NLP'

# Split according to label
df_trump = df[df['label'] == 0]
df_staff = df[df['label'] == 1]
df_trump = df_trump.drop(['id_num', 'name', 'dt'], axis=1)
df_staff = df_staff.drop(['id_num', 'name', 'dt'], axis=1)

# Train-valid split
df_trump_train, df_trump_valid = train_test_split(df_trump, train_size = 0.8, random_state = 1)
df_staff_train, df_staff_valid = train_test_split(df_staff, train_size = 0.8, random_state = 1)


# Concatenate splits of different labels
df_train = pd.concat([df_trump_train, df_staff_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_trump_valid, df_staff_valid], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv('train_forLSTM.csv', index=False)
df_valid.to_csv('valid_forLSTM.csv', index=False)

# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator

# Models

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import nltk
nltk.download('punkt')

# Fields
tokenizing = lambda x: nltk.tokenize.word_tokenize(x)
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(sequential=True, tokenize=tokenizing, lower=True, include_lengths=True, batch_first=True)
yr_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
fields = [('tweet', text_field), ('label', label_field)]

# TabularDataset
train_examp = [Example.fromlist([r.tweet, r.label], fields) for r in df.itertuples()]
train_data = Dataset(train_examp, fields=fields)
train, valid = Dataset.split(train_data, 0.8)
# train, valid = TabularDataset.splits(path=destination_folder, train='train.csv', validation='valid.csv',
                                           # format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.tweet),
                            device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.tweet),
                            device=device, sort=True, sort_within_batch=True)

# Vocabulary

text_field.build_vocab(train, min_freq=1, vectors='glove.twitter.27B.25d')

class LSTM(nn.Module):

    def __init__(self, dimension=300):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 25)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=25,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

import pickle
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
corp_vocab = text_field.vocab.itos
# print(len(text_field.vocab.itos))
new_embeddings = text_field.vocab.vectors
# print(new_embeddings)
newglove = dict(zip(corp_vocab, new_embeddings))
# print(newglove['hillary'])
# print(len(newglove.keys()))

with open('new embeddings.txt', 'w') as f:
    for word, vec in newglove.items():
        f.write('{} {}\n'.format(word, ' '.join(['{:e}'.format(item) for item in vec])))

# f = open(destination_folder + '/embeddings.pkl',"wb")
# pickle.dump(newglove, f)
# f.close()
# print(text_field.vocab.stoi)

# fine_trained_vectors = torchtext.vocab.Vectors(destination_folder + '/new embeddings.txt')
import torchtext.vocab as vocab

custom_embeddings = vocab.Vectors(name = 'new embeddings.txt')
text_field.build_vocab(train, min_freq=1, vectors=custom_embeddings)


# Training Function

def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=5,
          eval_every=len(train_iter) // 2,
          file_path=destination_folder,
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for ((tweet, tweet_len), labels), _ in train_loader:
            labels = labels.to(device)
            tweet = tweet.to(device)
            tweet_len = tweet_len.to(device)
            output = model(tweet, tweet_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for ((tweet, tweet_len), labels), _ in valid_loader:
                        labels = labels.to(device)
                        tweet = tweet.to(device)
                        tweet_len = tweet_len.to(device)
                        output = model(tweet, tweet_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint('LSTMmodel.pt', model, optimizer, best_valid_loss)
                    save_metrics('LSTMmetrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                    #pickle.dump(model, open('best_model.sav', 'wb'))
                    #torch.save(model.state_dict(), 'best_model.sav')
                    torch.save(model, 'best_model.sav')
                    best_model=model


    save_metrics('LSTMmetrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    print(train_loss_list)
    print(valid_loss_list)
    train_loss = train_loss_list
    valid_loss = valid_loss_list
    # x_axis = [x * 0.5 for x in range(1, len(train_loss) + 1)]
    # plt.plot(x_axis, train_loss, 'b-', label='Training loss')
    # plt.plot(x_axis, valid_loss, 'c-', label='Validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Loss')
    # plt.legend(loc="upper left")
    # plt.ylim(0, 2.0)
    # min_loss = min(valid_loss)
    # plt.annotate('min validation loss', xy=(x_axis[valid_loss.index(min_loss)], min_loss),
    #              xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.show()
    #trained_best_model = pickle.load(open('best_model.sav', 'rb'))
    return best_model


def LSTM_load_best_model():
  # Check which method works!
  #loaded_model = pickle.load(open('best_model.sav', 'rb'))

  # loaded_model = LSTM()
  # loaded_model.load_state_dict(torch.load('best_model.sav'))
  # loaded_model.eval()
  #
  loaded_model = torch.load('best_model.sav')
  return loaded_model

def LSTM_train_best_model():
  model = LSTM().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.003)
  m = train(model=model, optimizer=optimizer, num_epochs=3)
  evaluate(m, valid_iter)
  return m


def make_predictions(model, test_path, version='title', threshold=0.5):
    y_pred = []

    df = pd.read_csv(test_path, sep='\\t',
                     names=['name', 'tweet', 'dt'], engine='python')

    df = df.drop(['name', 'dt'], axis=1)
    df['tweet'] = df['tweet'].apply(lambda x: normalize_text(x))
    df = df.drop(df[(df.tweet == '')].index)

    tokenizing = lambda x: nltk.tokenize.word_tokenize(x)
    # label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(sequential=True, tokenize=tokenizing, lower=True, include_lengths=True, batch_first=True)
    fields = [('tweet', text_field)]

    test_examp = [Example.fromlist([r.tweet], fields) for r in df.itertuples()]
    test_data = Dataset(test_examp, fields=fields)

    test_iter = BucketIterator(test_data, batch_size=len(test_data), device=device)
    text_field.build_vocab(test_data, min_freq=1, vectors='glove.twitter.27B.25d')
    corp_vocab = text_field.vocab.itos
    # print(len(text_field.vocab.itos))
    new_embeddings = text_field.vocab.vectors
    # print(new_embeddings)
    newglove = dict(zip(corp_vocab, new_embeddings))
    # print(newglove['hillary'])
    # print(len(newglove.keys()))
    with open('new embeddings_test.txt', 'w') as f:
        for word, vec in newglove.items():
            f.write('{} {}\n'.format(word, ' '.join(['{:e}'.format(item) for item in vec])))
    custom_embeddings = vocab.Vectors(name='new embeddings_test.txt')
    text_field.build_vocab(test_data, min_freq=1, vectors=custom_embeddings)
    model.eval()

    with torch.no_grad():
        for ((tweet, tweet_len)), _ in test_iter:
            tweet = tweet.to(device)
            tweet_len = tweet_len.to(device)
            output = model(tweet, tweet_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())

    return y_pred



# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for ((tweet, tweet_len), labels), _ in test_loader:
            labels = labels.to(device)
            tweet = tweet.to(device)
            tweet_len = tweet_len.to(device)
            output = model(tweet, tweet_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    hm = sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    hm = hm.get_figure()
    hm.savefig("hm.png")
    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Staff', 'Trump'])
    ax.yaxis.set_ticklabels(['Staff', 'Trump'])

# best_model = LSTM().to(device)
# optimizer = optim.Adam(best_model.parameters(), lr=0.003)
# load_checkpoint('LSTMmodel.pt', best_model, optimizer)
# evaluate(best_model, valid_iter)


# import nltk
# nltk.download('punkt')
# tokenizing = lambda x: nltk.tokenize.word_tokenize(x)
# TEXT = Field(sequential=True, tokenize=tokenizing, lower=True)
# LABEL = Field(sequential=False, use_vocab=False)
# fields = [('tweet', TEXT), ('label', LABEL)]
# train_examp = [Example.fromlist([r.tweet, r.label], fields) for r in df.itertuples()]
# train = Dataset(train_examp, fields=fields)
# train_data, valid_data = Dataset.split(train, 0.8)
#
# TEXT.build_vocab(train_data, min_freq=1, vectors='glove.twitter.27B.25d')
# LABEL.build_vocab(train_data)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 64
# train_iterator, valid_iterator = BucketIterator.splits(
#     (train_data, valid_data),
#     batch_size = BATCH_SIZE,
#     sort_key = lambda x: len(x.tweet),
#     sort_within_batch=True,
#     device = device)
