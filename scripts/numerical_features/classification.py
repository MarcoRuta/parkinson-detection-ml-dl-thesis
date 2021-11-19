import warnings
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from termcolor import colored

warnings.filterwarnings( 'ignore' )

# I classificatori utilizzati
classifiers = [
    LogisticRegression( max_iter=500, solver="lbfgs" ),
    RandomForestClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    KNeighborsClassifier( 10 ),
    GaussianNB(),
    GradientBoostingClassifier( n_estimators=1000 ),
]

# I nomi dei classificatori utilizzati
names = [
    'Logistic Regression',
    'Random Forest',
    'SVC',
    'Decision Tree',
    'KNeighbors'
    'Gaussian Process',
    'Gradient Boosting',
]


# funzione che riceve come input il dataframe e stampa la correlazione di tutte le feature con l'attributo target
# e la matrice di correlazione di tutti gli attributi che sono correlati al target, ritorna la lista delle features correlate
def correlation(df):
    df.hist( bins=10, figsize=(20, 15), color='#E14906' )
    plt.tight_layout( pad=4 )
    plt.show()

    # Calcolo della correlazione di tutte le features con il target 'Y'
    corr = df.corr()['target']
    corr[np.argsort( corr, axis=0 )[::-1]]

    # Plot della correlazione di tutte le features con la feature 'Y'
    num_feat = df.columns[df.dtypes != object].drop( 'target' )
    features = num_feat[0:30]
    target = num_feat[0:29]
    labels = []
    values = []
    for col in features:
        labels.append( col )
        values.append( np.corrcoef( df[col].values, df.target.values )[0, 1] )

    ind = np.arange( len( labels ) )
    width = 0.9
    fig, ax = plt.subplots( figsize=(12, 8) )
    rects = ax.barh( ind, np.array( values ), color='green' )
    ax.set_yticks( ind + (width / 2.) )
    ax.set_yticklabels( labels, rotation='horizontal' )
    ax.set_xlabel( "Coefficente di relazione" )
    ax.set_title( "Correlazione con la feature target" )
    plt.show()

    related_features = ['no_strokes_st', 'magnitude_horz_vel_st', 'magnitude_vel_dy',
                        'magnitude_horz_vel_dy', 'magnitude_vert_vel_dy', 'magnitude_acc_dy',
                        'magnitude_horz_acc_dy', 'magnitude_vert_acc_dy', 'ncv_dy', 'nca_st', 'nca_dy', 'in_air_stcp',
                        'target']

    # Heatmap della correlazione tra tutti gli attributi
    corrMatrix = df[related_features].corr()
    sns.set( font_scale=1.10 )
    plt.figure( figsize=(10, 10) )
    sns.heatmap( corrMatrix, vmax=.8, linewidths=0.01,
                 square=True, annot=True, cmap='viridis', linecolor="white" )

    plt.title( 'Correlation between features' )
    plt.tight_layout( pad=4 )
    plt.show()

    return related_features


# funzione che calcola e restituisce l'accuracy di una predizione
def accuracy(prediction, actual):
    correct = 0
    not_correct = 0
    for i in range( len( prediction ) ):
        if prediction[i] == actual[i]:
            correct += 1
        else:
            not_correct += 1
    return (correct * 100) / (correct + not_correct)


# funzione che calcola e restituisce precision, recall e F1 di una predizione
def metrics(prediction, actual):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range( len( prediction ) ):
        if prediction[i] == actual[i] and actual[i] == 1:
            tp += 1
        if prediction[i] == actual[i] and actual[i] == 0:
            tn += 1
        if prediction[i] != actual[i] and actual[i] == 0:
            fp += 1
        if prediction[i] != actual[i] and actual[i] == 1:
            fn += 1
    metrics = {'Precision': (tp / (tp + fp + tn + fn)), 'Recall': (tp / (tp + fn)),
               'F1': (2 * (tp / (tp + fp + tn + fn)) * (tp / (tp + fn))) / (
                       (tp / (tp + fp + tn + fn)) + (tp / (tp + fn)))}
    return (metrics)


# funzione che carica i dati presenti in un file .csv e divide training e testing set
# la divisione è manuale dato che pazienti affetti e pazienti sani sono nello stesso csv
def data_load_and_split(path):
    # Lettura del dataset e salvataggio nel dataframe df
    df = pd.read_csv( path )

    related_features = correlation( df )

    # Divisione di training/testing set
    pos = df[df['target'] == 1]
    neg = df[df['target'] == 0]

    train_pos = pos.head( pos.shape[0] - 5 )
    train_neg = neg.head( pos.shape[0] - 5 )
    train = pd.concat( [train_pos, train_neg] )

    test_pos = pos.tail( 25 )
    test_neg = neg.tail( 15 )
    test = pd.concat( [test_pos, test_neg] )

    train_y = train['target']
    train_x = train[related_features]
    test_y = test['target']
    test_x = test[related_features]

    return df, train_x, train_y, test_x, test_y


# funzione che calcola ed inserisce in una tabella per ogni classificatore:
# matrice di confusione, accuracy, precision, recall, F1
# e attraverso 10-fold cross validation calcola la media di accuracy, precision, recall, F1
def classify(train_x, train_y, test_x, test_y):
    t = PrettyTable(
        ['Name', 'Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1', 'avg Accuracy', 'avg Precision',
         'avg Recall',
         'avg F1'] )

    for name, clf in zip( names, classifiers ):
        clf.fit( train_x, train_y )
        preds = clf.predict( test_x )
        _accuracy = accuracy( test_y.tolist(), preds.tolist() )
        _metrics = metrics( test_y.tolist(), preds.tolist() )

        _avg_accuracy = cross_val_score( clf, train_x, train_y, cv=10, scoring='accuracy' )
        _avg_precision = cross_val_score( clf, train_x, train_y, cv=10, scoring='precision_macro' )
        _avg_recall = cross_val_score( clf, train_x, train_y, cv=10, scoring='recall_macro' )
        _avg_F1 = cross_val_score( clf, train_x, train_y, cv=10, scoring='f1_macro' )

        predictions = cross_val_predict( clf, train_x, train_y, cv=10 )
        matrice = confusion_matrix( train_y, predictions )

        t.add_row(
            [colored( name, 'blue' ), matrice, round( _accuracy, 3 ), round( _metrics['Precision'], 3 ),
             round( _metrics['Recall'], 3 ), round( _metrics['F1'], 3 ),
             round( _avg_accuracy.mean(), 3 ), round( _avg_precision.mean(), 3 ), round( _avg_recall.mean(), 3 ),
             round( _avg_F1.mean(), 3 )] )

        t.add_row( ['', '', '', '', '', '', '', '', '', ''] )

    print( t )


if __name__ == '__main__':
    # come path va fornito quello del file extracted_data.csv
    df, train_x, train_y, test_x, test_y = data_load_and_split(
        'E:/Desktop/Parkinson_py/dataset/numerical_dataset/extracted_data.csv' )
    classify( train_x, train_y, test_x, test_y )
