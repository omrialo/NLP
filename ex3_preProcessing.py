from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import copy
from torchnlp.word_to_vector import GloVe
from sklearn.decomposition import PCA
import sklearn.preprocessing
import nltk #import word_tokenize, pos_tag
import numpy as np



def getTfidfFFeatures(train, test, tfidfParams):
    """ extract TF-IDF features
        Args:
     	  train (df): train data frame
     	  test (df): train data frame
     	  tfidfParams (dict): dict of the TF-IDF  parameters from config

    Returns:
        train (df): train data frame after features added
     	test (df): train data frame features added
    """
    for index, row in train.iterrows():
        train.loc[index, '<tweet text>'] = normalize_text(row['<tweet text>'])
    for index, row in test.iterrows():
        test.loc[index, '<tweet text>'] = normalize_text(row['<tweet text>'])
    stop = set(stopwords.words('english'))
    tfidf = TfidfVectorizer(encoding='utf - 8', lowercase=False, analyzer='word', stop_words=stop,
                            max_df=tfidfParams['max_df'], min_df=tfidfParams['min_df'])
    tfidfTrain = tfidf.fit_transform(train['<tweet text>'])
    tfidfTest = tfidf.transform(test['<tweet text>'])

    trainFeaturesDF = pd.DataFrame(tfidfTrain.todense(), columns=tfidf.get_feature_names())
    #train = pd.concat([train, trainFeaturesDF], axis=1)
    # train = train.join(trainFeaturesDF)

    testFeaturesDF = pd.DataFrame(tfidfTest.todense(), columns=tfidf.get_feature_names())
    test = test.reset_index(drop=True)
    testFeaturesDF = testFeaturesDF.reset_index(drop=True)

    # test = test.join(testFeaturesDF)

    #test = pd.concat([test, testFeaturesDF], axis=1)
    train = train.drop('<tweet text>', 1)
    test = test.drop('<tweet text>', 1)
    train = train.drop('<tweet id>', 1)
    train = train.drop('<user handle>', 1)
    test = test.drop('<user handle>', 1)
    train = train.drop('<time stamp>', 1)
    test = test.drop('<time stamp>', 1)
    return train, test


def createLabel(train):
    """ creating the dependent variable
        Args:
     	  train (df): train data frame

    Returns:
        train (df): train data frame with the dependent variable
    """
    # creating the dependent variable:
    train = train[(train['<device> = Label'] == "android") | (train['<device> = Label'] == "iphone")]
    train = copy.deepcopy(train)
    tempTrain = pd.get_dummies(train['<device> = Label'], prefix='<device> = Label')
    train['isStaffer'] = copy.deepcopy(tempTrain['<device> = Label_iphone'])
    # train['isStaffer'] = train['<device> = Label'].apply(lambda x: 0 if x == 'android' else 1)
    train = train.drop('<device> = Label', 1)
    return train

def extractTimeBaseFeatures(train, test):
    """ extract timebase features
        Args:
     	  train (df): train data frame
     	  test (df): train data frame

    Returns:
        train (df): train data frame after features added
     	test (df): train data frame features added
    """

    train['<time stamp>'] = pd.to_datetime(train['<time stamp>'])
    train['hr'] = train['<time stamp>'].dt.hour
    train['d'] = train['<time stamp>'].dt.weekday
    train['yr'] = train['<time stamp>'].dt.year
    train['mo'] = train['<time stamp>'].dt.month

    test['<time stamp>'] = pd.to_datetime(test['<time stamp>'])
    test['hr'] = test['<time stamp>'].dt.hour
    test['d'] = test['<time stamp>'].dt.weekday
    test['yr'] = test['<time stamp>'].dt.year
    test['mo'] = test['<time stamp>'].dt.month


    cols = ['<time stamp>', 'hr', 'd', 'yr', 'mo']

    for col in cols:
        scaler = MinMaxScaler()
        scaler.fit(copy.deepcopy(train[[col]]))
        train[col] = copy.deepcopy(scaler.transform(train[[col]]))
        test[col] = copy.deepcopy(scaler.transform(test[[col]]))

    return train, test


def token_to_pos(ch):
    tokens = nltk.word_tokenize(ch)
    return [p[1] for p in nltk.pos_tag(tokens)]



def extractTrumpUniqueFeatures(train, test):
    """ extract Trump unique tweet features
        Args:
     	  train (df): train data frame
     	  test (df): train data frame

    Returns:
        train (df): train data frame after features added
     	test (df): train data frame features added
    """
    # state_names = ["Alaska", "Alabama", "Arkansas", "Hillary","Obama", "California", "Colorado",
    #                "Connecticut", "District", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii",
    #                "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
    #                "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina",
    #                "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York",
    #                "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina",
    #                "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington",
    #                "Wisconsin", "West Virginia", "Wyoming", "Las Vegas", "(!)", '!!!', '!!', '??', '???'
    #                ]
    tweetLength = []
    capitalCount = []
    hashtagsCount = []
    retweetCount = []
    persentagelCount = []
    spacesCount = []
    upperWordsCount = []
    uniqueChars = []
    DT=[]
    IN=[]
    JJ=[]
    NNS=[]
    NN = []


    for tweet in train['<tweet text>']:
        # number of capital letters:
        capitalCount.append(sum(1 for c in tweet if c.isupper()))

        tweetLength.append(len(tweet))

        counter = 0
        for c in tweet:
            if c == '-' or c == "'" or c == '!' or c == '?' or c == '/' or c == '$':
                counter += 1
        uniqueChars.append(counter )#+ (len(tweet.split(" ") - len([word for word in tweet.split(" ") if glove.is_include(word)]))))

        counter = 0
        tokens = normalize_text(tweet).split(' ')
        chapters_pos = [token_to_pos(token) for token in tokens]
        # count frequencies for common POS types
        pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']

        fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                                   for ch in chapters_pos]).astype(np.float64)
        fvs_syntax = pd.DataFrame(
            {'NN': fvs_syntax[:, 0], 'NNP': fvs_syntax[:, 1], 'DT': fvs_syntax[:, 2],
             'IN': fvs_syntax[:, 3], 'JJ': fvs_syntax[:, 4], 'NNS': fvs_syntax[:, 5]})
        # normalise by dividing each row by number of tokens in the chapter
        fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
        NN.append(sum(fvs_syntax['NN']))
        DT.append(sum(fvs_syntax['DT']))
        IN.append(sum(fvs_syntax['IN']))
        JJ.append(sum(fvs_syntax['JJ']))
        NNS.append(sum(fvs_syntax['NNS']))


        upperWordsCount.append(sum(map(str.isupper, tweet.split())))
        # number of hashtags:
        hashtagsCount.append(len(tweet.split('#')))

        # check retweet:
        retweetCount.append(len(tweet.split('@realDonaldTrump')))

        # count how many time his favorite new channel is mention:
        persentagelCount.append(len(tweet.split('%')))

        # count spaces:
        spacesCount.append(len(tweet.split(' ')))

    #train['isWord'] = isWord
    train['tweetLength'] = tweetLength
    train['upperWordsCount'] = upperWordsCount
    train['capitalLetters'] = capitalCount
    train['hashtags'] = hashtagsCount
    train['retweet'] = retweetCount
    train['persentagelCount'] = persentagelCount
    train['spaced'] = spacesCount
    train['uniqueChars'] = uniqueChars
    train['NN'] = NN
    train['IN'] = IN
    train['JJ'] = JJ
    train['NNS'] = NNS
    train['DT'] = DT

    capitalCount = []
    hashtagsCount = []
    retweetCount = []
    persentagelCount = []
    spacesCount = []
    upperWordsCount = []
    tweetLength = []
    #isWord = []
    uniqueChars = []
    DT=[]
    IN=[]
    JJ=[]
    NNS=[]
    NN = []


    for tweet in test['<tweet text>']:
        # number of capital letters:
        capitalCount.append(sum(1 for c in tweet if c.isupper()))

        counter = 0
        for c in tweet:
            if c == '-' or c == "'" or c == '!' or c == '?' or c == '/' or c == '$':
                counter += 1
        uniqueChars.append(counter)#+ (len(tweet.split(" ") - len([word for word in tweet.split(" ") if glove.is_include(word)]))))

        tokens = normalize_text(tweet).split(' ')
        chapters_pos = [token_to_pos(token) for token in tokens]
        # count frequencies for common POS types
        pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']

        fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                                   for ch in chapters_pos]).astype(np.float64)
        fvs_syntax = pd.DataFrame(
            {'NN': fvs_syntax[:, 0], 'NNP': fvs_syntax[:, 1], 'DT': fvs_syntax[:, 2],
             'IN': fvs_syntax[:, 3], 'JJ': fvs_syntax[:, 4], 'NNS': fvs_syntax[:, 5]})
        # normalise by dividing each row by number of tokens in the chapter
        fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
        NN.append(sum(fvs_syntax['NN']))
        DT.append(sum(fvs_syntax['DT']))
        IN.append(sum(fvs_syntax['IN']))
        JJ.append(sum(fvs_syntax['JJ']))
        NNS.append(sum(fvs_syntax['NNS']))


        tweetLength.append(len(tweet))
        # number of hashtags:
        hashtagsCount.append(len(tweet.split('#')))

        upperWordsCount.append(sum(map(str.isupper, tweet.split())))

        # check retweet:
        retweetCount.append(len(tweet.split('@realDonaldTrump')))

        # count how many time his favorite new channel is mention:
        persentagelCount.append(len(tweet.split('%')))

        # count spaces:
        spacesCount.append(len(tweet.split(' ')))

    test['uniqueChars'] = uniqueChars
    test['NN'] = NN
    test['DT'] = DT
    test['IN'] = IN
    test['JJ'] = JJ
    test['NNS'] = NNS
    test['upperWordsCount'] = upperWordsCount
    test['capitalLetters'] = capitalCount
    test['hashtags'] = hashtagsCount
    test['retweet'] = retweetCount
    test['persentagelCount'] = persentagelCount
    test['spaced'] = spacesCount
    test['tweetLength'] = tweetLength

    cols = ['capitalLetters', 'hashtags', 'retweet', 'persentagelCount','upperWordsCount', 'spaced', 'tweetLength','uniqueChars','NN', 'DT', 'IN', 'JJ', 'NNS']

    for col in cols:
        scaler = MinMaxScaler()
        scaler.fit(copy.deepcopy(train[[col]]))
        train[col] = copy.deepcopy(scaler.transform(train[[col]]))
        test[col] = copy.deepcopy(scaler.transform(test[[col]]))

    return train, test



def extractEmbeddingsFeatures(train_data, test_data, perform_pca=False, pca_dim=5):
    train_data.to_csv('checkTrain.csv', index=False)
    """ extract timebase features
        Args:
     	  train (df): train data frame
     	  test (df): train data frame

    Returns:
        train (df): train data frame after features added
     	test (df): train data frame features added
    """
    if train_data.shape[1] > 2:
        train_data = train_data.drop('<tweet id>', 1)
        train_data = train_data.drop('<user handle>', 1)
        train_data = train_data.drop('<time stamp>', 1)
        test_data = test_data.drop('<user handle>', 1)
        test_data = test_data.drop('<time stamp>', 1)

    for index, row in train_data.iterrows():
        train_data.loc[index, '<tweet text>'] = normalize_text_forEmbeddings(row['<tweet text>'])
    for index, row in test_data.iterrows():
        test_data.loc[index, '<tweet text>'] = normalize_text_forEmbeddings(row['<tweet text>'])

    corpus = ""
    max_len = 0
    embedding_dim = 300
    #name='twitter.27B', dim=embedding_dim
    glove = GloVe()
    for index, row in train_data.iterrows():
        corpus += row['<tweet text>']
        row_twit_tokens = row['<tweet text>'].split(' ')
        if "" in row_twit_tokens:
            row_twit_tokens.remove("")
        max_len = max(max_len, len(row_twit_tokens))  # check if its a good tokenizer !!!!!!!!!!!!!!!!!!!!!!!!

    for index, row in test_data.iterrows():
        corpus += row['<tweet text>']
        row_twit_tokens = row['<tweet text>'].split(' ')
        if "" in row_twit_tokens:
            row_twit_tokens.remove("")
        max_len = max(max_len, len(row_twit_tokens))  # check if its a good tokenizer !!!!!!!!!!!!!!!!!!!!!!!!

    lis = []
    idx = 0
    new_train_data = [[0] * max_len * embedding_dim for i in range(len(train_data))]
    #words_matrix = [[0] * max_len for i in range(len(train_data))]
    for index, row in train_data.iterrows():
        row_twit_tokens = row['<tweet text>'].split(' ')
        if "" in row_twit_tokens:
            row_twit_tokens.remove("")
        twit = (row['<tweet text>'])
        for i in range(max_len):
            if i < (max_len - len(row_twit_tokens)):
                continue
            else:
                lis = glove[row_twit_tokens[(i + len(row_twit_tokens) - max_len)]].tolist()
                #words_matrix[index][i] = row_twit_tokens[(i + len(row_twit_tokens) - max_len)]
                for j in range(embedding_dim):
                    if lis == [0] * embedding_dim:
                        new_train_data[idx][j + (embedding_dim * i)] = 1
                    else:
                        new_train_data[idx][j + (embedding_dim * i)] = lis[j]
        idx += 1

    idx = 0
    lis = []
    new_test_data = [[0] * max_len * embedding_dim for i in range(len(test_data))]
    #words_matrix = [[0] * max_len for i in range(len(test_data))]
    for index, row in test_data.iterrows():
        row_twit_tokens = row['<tweet text>'].split(' ')
        if "" in row_twit_tokens:
            row_twit_tokens.remove("")
        twit = (row['<tweet text>'])
        for i in range(max_len):
            if i < (max_len - len(row_twit_tokens)):
                continue
            else:
                lis = glove[row_twit_tokens[(i + len(row_twit_tokens) - max_len)]].tolist()
                #words_matrix[index][i] = row_twit_tokens[(i + len(row_twit_tokens) - max_len)]
                if lis == [0] * embedding_dim:
                    new_test_data[idx][j + (embedding_dim * i)] = 1
                else:
                    new_test_data[idx][j + (embedding_dim * i)] = lis[j]
        idx += 1

    #words_matrix = pd.DataFrame(words_matrix)
    #words_matrix.to_csv('embeddings_words2.csv', index=False)

    if perform_pca:
        pca = PCA(n_components=pca_dim)
        new_train_data = pca.fit_transform(new_train_data)
        new_test_data = pca.fit_transform(new_test_data)
        new_train_data = sklearn.preprocessing.normalize(new_train_data)
        new_test_data = sklearn.preprocessing.normalize(new_test_data)


    new_train_data = pd.DataFrame(new_train_data)
    new_test_data = pd.DataFrame(new_test_data)
    train_data = train_data.join(new_train_data)
    test_data = test_data.join(new_test_data)
    train_data.to_csv('embeddings_full_train.csv', index=False)

    # train_data = train_data.drop('<tweet text>', 1)
    # test_data = test_data.drop('<tweet text>', 1)
    return train_data, test_data


def normalize_text(text):
    # Remove all characters that are not letters or ends of sentences, remove multiple spaces, change to lower case
    text = re.sub(r'[^A-Za-z ]', '', text)  # Remove unnecessary characters
    # text = re.sub('([.!?])', r' \1 ', text)  # Pad ends od sentences with spaces
    text = re.sub(' +', ' ', text)  # Remove multiple spaces
    text = text.lower()  # Switch all to lower case
    text = text.split('http', 1)[0]

    # Remove excess spaces at the edges and return
    return text.strip()

def normalize_text_forEmbeddings(text):
    # Remove all characters that are not letters or ends of sentences, remove multiple spaces, change to lower case
    text = re.sub(r'[^A-Za-z ]', '', text)  # Remove unnecessary characters
    # text = re.sub('([.!?])', r' \1 ', text)  # Pad ends od sentences with spaces
    text = re.sub(' +', ' ', text)  # Remove multiple spaces
    text = text.lower()  # Switch all to lower case
    text = text.split('http', 1)[0]

    # Remove excess spaces at the edges and return
    return text.strip()