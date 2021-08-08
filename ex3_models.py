import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc,precision_score,recall_score,roc_curve,f1_score
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from config import NeuralNetworkParams
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class NeuralNetwork():
    input_dim = 0
    hidden_dim = []
    output_dim = 0
    model = ""
    def __init__(self, input_dim, hidden_dim, output_dim, convolutional=False):
        # Hyperparameters for our network
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Build a feed-forward network
        self.model = self.build_network(convolutional)

    def build_network(self, convolutional):
        if convolutional == False:
            model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim[0]),
                                  nn.Tanh(),
                                  nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
                                  nn.Tanh(),
                                  nn.Linear(self.hidden_dim[1], self.hidden_dim[2]),
                                  nn.Tanh(),
                                  # nn.Linear(self.hidden_dim[2], self.hidden_dim[3]),
                                  # nn.Sigmoid(),
                                  #nn.Dropout(),
                                  nn.Linear(self.hidden_dim[2], self.output_dim),
                                  nn.Softmax(dim=1))
        else:
            print(0)
        return model

    def train_network(self, features, labels):
        """ trains NN model
            Args:
                features(df): training features data frame
                labels(df): labels df
        """
        torch.manual_seed(89)
        X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.1,
                                                                        random_state=42)
        features = torch.tensor(X_train.values)
        labels = torch.tensor(y_train.values)
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=0.0003)
        epochs = NeuralNetworkParams['epochs']
        batches = NeuralNetworkParams['batch_size']
        #trainloader = torch.utils.data.DataLoader(Dataset(features, labels), batch_size=NeuralNetworkParams['batch_size'], shuffle=True)
        for e in range(epochs):
            running_loss = 0
            for i in range(batches):
                # Local batches and labels
                local_X, local_y = features[i * batches:(i + 1) * batches, ], labels[i * batches:(i + 1) * batches, ]

                # Training pass
                optimizer.zero_grad()

                output = self.model(local_X.float())
                loss = criterion(output, local_y.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print("Training loss:", str(running_loss))

        pred = self.model(torch.tensor(X_validation.values).float())

        predicted = torch.argmax(pred.data, dim=1)
        true_class = torch.tensor(y_validation.values)
        train_acc = torch.sum(predicted == true_class)
        print(X_validation.dtypes)
        print("NN Accuracy:", str(train_acc.item()/len(y_validation)))
        print(predicted)
        print(y_validation)
        print(y_validation.dtypes)
        print("NN precision:", str(precision_score(y_validation.values.ravel(), predicted.numpy())))
        print("NN recall:", str(recall_score(y_validation.values.ravel(), predicted.numpy())))
        fpr, tpr, thresholds = roc_curve(y_validation.values.ravel(), predicted.numpy())
        print("NN auc:", str(auc(fpr, tpr)))
        print("NN f1:", str(f1_score(y_validation.values.ravel(), predicted.numpy())))


def train_RandomForest(train_data, n_estimators=300):
    """ train Random Forest model
        Args:
       n_estimators (int): number of trees

        Returns:
        model: trained model
    """
    X = train_data.drop('isStaffer', 1)
    Y = train_data[['isStaffer']]
    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None,
    ccp_alpha=0.0, max_samples=None)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    results = cross_val_score(model, X, Y.values.ravel(), cv=kfold)
    print("train_RandomForest Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    model = model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_validation)
    print("train_RandomForest precision:", str(precision_score(y_validation.values.ravel(), pred)))
    print("train_RandomForest recall:", str(recall_score(y_validation.values.ravel(), pred)))
    fpr, tpr, thresholds = roc_curve(y_validation.values.ravel(), pred)
    print("train_RandomForest auc:", str(auc(fpr, tpr)))
    print("train_RandomForest f1:", str(f1_score(y_validation.values.ravel(), pred)))
    #print(model.feature_importances_)
    # estimator = model.estimators_[5]

    # from sklearn.tree import export_graphviz
    # # Export as dot file
    # export_graphviz(estimator, out_file='tree.dot',
    #                 feature_names=list(X.columns),
    #                 rounded=True, proportion=False,
    #                 precision=2, filled=True)
    #
    # # Convert to png using system command (requires Graphviz)
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename='tree.png')
    return model


def train_LogisticRegression(train_data, solver, penalty, C):
    """ train Logistic Regression model
        Args:
       solver (str): solving algorithm name
       penalty (str): regularization technique
       C (float): regularization parameter

        Returns:
        model: trained model
    """
    X = train_data.drop('isStaffer', 1)
    Y = train_data[['isStaffer']]
    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1,
                            class_weight=None, random_state=None, solver=solver, max_iter=10000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    #model = lr.fit(X_train, y_train)
    # pred = model.predict(X_validation)
    # print("train_LogisticRegressionfunction Accuracy:", str(accuracy_score(y_validation.to_numpy(dtype=float), pred)))
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    results = cross_val_score(model, X, Y.values.ravel(), cv=kfold)
    print("train_LogisticRegression Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #model = model.fit(X, Y.values.ravel())
    #pred = model.predict(X_validation)
    #print("train_LogisticRegressionfunction Accuracy:", str(accuracy_score(y_validation.values.ravel(), pred)))
    model = model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_validation)
    print("train_LogisticRegression precision:", str(precision_score(y_validation.values.ravel(), pred)))
    print("train_LogisticRegression recall:", str(recall_score(y_validation.values.ravel(), pred)))
    fpr, tpr, thresholds = roc_curve(y_validation.values.ravel(), pred)
    print("train_LogisticRegression auc:", str(auc(fpr, tpr)))
    print("train_LogisticRegression f1:", str(f1_score(y_validation.values.ravel(), pred)))
    return model

def train_SVM(train_data, C, kernel, gamma='scale'):
    """ train SVM model
        Args:
       kernel (str): kernel name
       gamma (str or float): regularization technique
       C (float): regularization parameter

        Returns:
        model: trained model
    """
    X = train_data.drop('isStaffer', 1)
    Y = train_data[['isStaffer']]
    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2,
                                                                    random_state=42)
    model = sklearn.svm.SVC(C=C, kernel=kernel,gamma=gamma)
    # model = clf.fit(X_train, y_train) # train the data - fit the model per each class binary classifier
    # pred = model.predict(X_validation)
    # print(kernel, "train_LinearSVMfunction Accuracy:", str(accuracy_score(y_validation.to_numpy(dtype=float), pred)))
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    results = cross_val_score(model, X, Y.values.ravel(), cv=kfold)
    print(kernel, "train_SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # model = model.fit(X, Y.values.ravel())
    model = model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_validation)
    print("train_SVM precision:", str(precision_score(y_validation.values.ravel(), pred)))
    print("train_SVM recall:", str(recall_score(y_validation.values.ravel(), pred)))
    fpr, tpr, thresholds = roc_curve(y_validation.values.ravel(), pred)
    print("train_SVM auc:", str(auc(fpr, tpr)))
    print("train_SVM f1:", str(f1_score(y_validation.values.ravel(), pred)))
    return model


