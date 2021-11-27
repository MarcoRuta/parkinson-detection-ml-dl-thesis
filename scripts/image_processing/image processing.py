import glob
import uuid

import numpy
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from skimage.exposure import exposure
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored


# I classificatori utilizzati
classifiers = [
    LogisticRegression( max_iter=500, solver="lbfgs" ),
    RandomForestClassifier( max_features=3, max_leaf_nodes=3 ),
    SVC(),
    DecisionTreeClassifier(),
    KNeighborsClassifier( 5 ),
    GradientBoostingClassifier( n_estimators=128 ),
]


# I nomi dei classificatori utilizzati
names = [
    'Logistic Regression',
    'SVC',
    'Random Forest',
    'Decision Tree',
    'KNeighbors',
    'Gradient Boosting',
]


# Metodo che esegue data augmentation, per ogni immagine in in_path vengono generate 50 variazioni in out_path
def data_augmentation(in_path, out_path):
    print( "Data augmentation..." )
    for filename in glob.glob( in_path ):

        image = load_img( filename )
        image = img_to_array( image )
        image = np.expand_dims( image, axis=0 )

        # inizializzazione parametri generator utilizzato per data augmentation
        aug = ImageDataGenerator( rotation_range=30, width_shift_range=0.1,
                                  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                  horizontal_flip=True, fill_mode="nearest" )
        total = 0

        # lista di immagini autogenerate
        imageGen = aug.flow( image, batch_size=1, save_to_dir=out_path,
                             save_prefix=uuid.uuid4(), save_format="jpg" )

        for image in imageGen:
            total += 1

            # per ogni immagine ne vengono generate 50
            if total == 50:
                break


# Estrae l'istrogramma dei gradienti orientati data un immagine come input
def quantify_image(image):

    features = feature.hog( image, orientations=9,
                            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1" )

    # ritorna il vettore delle features
    return features


# carica le foto da una directory, ne estrae le features e recupera il target dal nome della directory
def load_split(path):

    imagePaths = list( paths.list_images( path ) )
    data = []
    labels = []


    for imagePath in imagePaths:
        #viene estratta la label dalla folder di appartenenza
        label = imagePath.split( os.path.sep )[-2]

        # viene caricata l'immagine, convertita in bianco e nero e scalata
        image = cv2.imread( imagePath )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
        image = cv2.resize( image, (200, 200) )

        # funzione di threshold per isolare l'immagine
        image = cv2.threshold( image, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )[1]

        # estrazione dell' HOG dall'immagine
        features = quantify_image( image )

        data.append( features )
        labels.append( label )

    # return dati e labels
    return (np.array( data ), np.array( labels ))


# genera e mostra l'istogramma dei gradienti orientati per una generica immagine del dataset
def plot_hog():
    image = Image.open( 'V01HE01.png' )

    # creating hog features
    fd, hog_image = feature.hog( image, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualize=True, multichannel=True )

    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(8, 4), sharex=True, sharey=True )

    ax1.axis( 'off' )
    ax1.imshow( image, cmap=plt.cm.gray )
    ax1.set_title( 'Input image' )

    hog_image_rescaled = exposure.rescale_intensity( hog_image, in_range=(0, 10) )

    ax2.axis( 'off' )
    ax2.imshow( hog_image_rescaled, cmap=plt.cm.gray )
    ax2.set_title( 'Istogramma dei gradienti (HOG)' )
    plt.show()


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


# dati i path di training e test recupera trainX trainY testX e testY
def load_and_split_data(trainingPath, testingPath):
    # define the path to the training and testing directories

    # loading the training and testing data
    print( "[INFO] loading data..." )
    (trainX, trainY) = load_split( trainingPath )
    (testX, testY) = load_split( testingPath )

    # encode the labels as integers
    le = LabelEncoder()
    trainY = le.fit_transform( trainY )
    testY = le.transform( testY )

    return trainX, trainY, testX, testY


# funzione che calcola ed inserisce in una tabella per ogni classificatore:
# matrice di confusione, accuracy, precision, recall, F1
# e attraverso 10-fold cross validation calcola la media di accuracy, precision, recall, F1
def classify(train_x, train_y, test_x, test_y):
    t = PrettyTable(
        ['Name', 'Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1', 'avg Accuracy', 'avg Precision',
         'avg Recall',
         'avg F1'] )


    for name, clf in zip( names, classifiers ):
        print("train e validation")
        clf.fit( train_x, train_y )
        preds = clf.predict( test_x )
        _accuracy = accuracy( test_y.tolist(), preds.tolist() )
        _metrics = metrics( test_y.tolist(), preds.tolist() )

        x = numpy.concatenate( [train_x, test_x] )
        y = numpy.concatenate( [train_y, test_y] )

        print( "cross" )
        _avg_accuracy = cross_val_score( clf, x, y, cv=5, scoring='accuracy' )
        _avg_precision = cross_val_score( clf, x, y, cv=5, scoring='precision_macro' )
        _avg_recall = cross_val_score( clf, x, y, cv=5, scoring='recall_macro' )
        _avg_F1 = cross_val_score( clf, x, y, cv=5, scoring='f1_macro' )

        predictions = cross_val_predict( clf, x, y, cv=5 )
        matrice = confusion_matrix( y, predictions )


        t.add_row(
            [colored( name, 'blue' ), matrice, round( _accuracy, 3 ), round( _metrics['Precision'], 3 ),
             round( _metrics['Recall'], 3 ), round( _metrics['F1'], 3 ),
             round( _avg_accuracy.mean(), 3 ), round( _avg_precision.mean(), 3 ), round( _avg_recall.mean(), 3 ),
             round( _avg_F1.mean(), 3 )] )
        print( "riga aggiunta" )
        t.add_row( ['', '', '', '', '', '', '', '', '', ''] )

    print( t )


# pulisce le directory che sono state riempite con data_augmentation
def clear_data():
    files = glob.glob( 'E:/Desktop/Parkinson_py/dataset/augmented_dataset/training/parkinson/*' )
    for f in files:
        os.remove( f )

    files = glob.glob( 'E:/Desktop/Parkinson_py/dataset/augmented_dataset/training/healthy/*' )
    for f in files:
        os.remove( f )

    files = glob.glob( 'E:/Desktop/Parkinson_py/dataset/augmented_dataset/testing/parkinson/*' )
    for f in files:
        os.remove( f )

    files = glob.glob( 'E:/Desktop/Parkinson_py/dataset/augmented_dataset/testing/healthy/*' )
    for f in files:
        os.remove( f )


if __name__ == '__main__':


    clear_data()

    data_augmentation( 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/training/parkinson/*',
                       'E:/Desktop/Parkinson_py/dataset/augmented_dataset/training/parkinson' )
    data_augmentation( 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/training/healthy/*',
                       'E:/Desktop/Parkinson_py/dataset/augmented_dataset/training/healthy' )
    data_augmentation( 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/testing/parkinson/*',
                       'E:/Desktop/Parkinson_py/dataset/augmented_dataset/testing/parkinson' )
    data_augmentation( 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/testing/healthy/*',
                       'E:/Desktop/Parkinson_py/dataset/augmented_dataset/testing/healthy' )

    trainX, trainY, testX, testY = load_and_split_data( 'E:/Desktop/Parkinson_py/dataset/augmented_dataset/training',
                                                        'E:/Desktop/Parkinson_py/dataset/augmented_dataset/testing' )

    #plot_hog()

    classify( trainX, trainY, testX, testY )

    # cross_val(trainX, trainY)

    clear_data()
