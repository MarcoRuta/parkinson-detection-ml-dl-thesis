import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from skimage import feature
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc

# data una generica immagine ne estrae l'istogramma dei gradienti orientati
def quantify_image(image):
    features = feature.hog( image, orientations=9,
                            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1" )

    # ritorna il vettore delle features
    return features


def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list( paths.list_images( path ) )
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split( os.path.sep )[-2]

        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread( imagePath )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
        image = cv2.resize( image, (200, 200) )

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold( image, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )[1]

        # quantify the image
        features = quantify_image( image )

        # update the data and labels lists, respectively
        data.append( features )
        labels.append( label )

    # return the data and labels
    return (np.array( data ), np.array( labels ))

def accuracy(prediction, actual):
    correct = 0
    not_correct = 0
    for i in range( len( prediction ) ):
        if prediction[i] == actual[i]:
            correct += 1
        else:
            not_correct += 1
    return (correct * 100) / (correct + not_correct)


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


def cnn_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, model_params['img_width'], model_params['img_height'])
    else:
        input_shape = (model_params['img_width'], model_params['img_height'], 3)

    model = Sequential()

    for i, num_filters in enumerate( model_params['filters'] ):
        if i == 0:
            model.add( Conv2D( num_filters, (3, 3), input_shape=input_shape ) )
        else:
            model.add( Conv2D( num_filters, (3, 3) ) )
        model.add( Activation( 'relu' ) )
        model.add( MaxPooling2D( pool_size=(2, 2) ) )

    model.add( Flatten() )
    model.add( Dense( 500 ) )
    model.add( Activation( 'relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense( 1 ) )  # single output neuron (output ranged from 0-1; binary class)
    model.add( Activation( 'sigmoid' ) )
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model


def data_augmentation():
    train_datagen = ImageDataGenerator(
        zoom_range=model_params['zoom_range'],
        height_shift_range=0.1,
        rotation_range=model_params['rotation_range'],
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=model_params['horizontal_flip']
    )

    validation_datagen = ImageDataGenerator( rescale=1. / 255 )

    train_generator = train_datagen.flow_from_directory(
        model_params['train_data_dir'],
        target_size=(model_params['img_width'], model_params['img_height']),
        batch_size=24,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        model_params['validation_data_dir'],
        target_size=(model_params['img_width'], model_params['img_height']),
        batch_size=30,
        class_mode='binary',
        shuffle=False  # Change to False if using plot_roc
    )

    return (train_datagen, validation_datagen, train_generator, validation_generator)


def plot_roc():
    validation_generator.reset()
    x, classes = next( validation_generator )
    preds = model.predict( x, verbose=1 )

    fpr, tpr, _ = roc_curve( classes, preds )
    roc_auc = auc( fpr, tpr )
    plt.figure()
    lw = 2
    plt.plot( fpr, tpr, color='darkturquoise', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc )
    plt.plot( [0, 1], [0, 1], color='navy', lw=lw, linestyle='--' )
    plt.xlim( [0.0, 1.0] )
    plt.ylim( [0.0, 1.05] )
    plt.xlabel( 'False Positive Rate' )
    plt.ylabel( 'True Positive Rate' )
    plt.title( 'Spiral Model Receiver operating characteristic' )
    plt.legend( loc="lower right" )
    plt.show()


def model_evaluation_plot():
    plt.plot( history.history['accuracy'], label='accuracy' )
    plt.plot( history.history['val_accuracy'], label='val_accuracy' )
    plt.plot( history.history['loss'], label='loss' )
    plt.plot( history.history['val_loss'], label='val_loss' )
    plt.xlabel( 'Epoch' )
    plt.ylim( [0, 1] )
    plt.legend( loc='lower right' )
    plt.show()


if __name__ == '__main__':

    spiral_params = {
        'filters': [32, 32, 32, 32, 64],
        'img_width': 256,
        'img_height': 256,
        'train_data_dir': 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/training',
        'validation_data_dir': 'E:/Desktop/Parkinson_py/dataset/image_dataset/spiral/testing',
        'epochs': 800,
        'batch_size': 24,
        'zoom_range': 0.1,
        'rotation_range': 360,
        'horizontal_flip': False,
        'weights': 'spiral_weights.h5'
    }

    model_params = spiral_params

    model = cnn_model()

    train_datagen, validation_datagen, train_generator, validation_generator = data_augmentation()

    # decommentare se si vogliono caricare gli ultimi pesi
    #model.load_weights(model_params['weights'])

    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=3,
        epochs=model_params['epochs'],
        validation_data=validation_generator,
        validation_steps=1
    )

    model.evaluate( validation_generator )

    plot_roc() # Change validation shuffle to False if using plot_roc

    model_evaluation_plot()

    model.save_weights('spiral_weights.h5')

