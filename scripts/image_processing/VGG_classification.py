import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, Sequential
from keras.applications import vgg16
from keras.layers import Flatten, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve, auc


def VGG_model():

    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3) )

    for layer in base_model.layers[:-8]:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add( Flatten() )
    model.add( Dense( 500 ) )
    model.add( Activation( 'relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense( 1 ) )  # single output neuron (output ranged from 0-1; binary class)
    model.add( Activation( 'sigmoid' ) )


    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.SGD( learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


def data_augmentation():
    Train_Datagen = ImageDataGenerator( dtype='float32',
                                        preprocessing_function=tf.keras.applications.resnet.preprocess_input )
    Val_Datagen = ImageDataGenerator( dtype='float32',
                                      preprocessing_function=tf.keras.applications.resnet.preprocess_input )

    train_gen = Train_Datagen.flow_from_directory( directory=spiral_params['train_data_dir'], target_size=(256, 256),
                                                   batch_size=24, class_mode='binary' )

    val_gen = Val_Datagen.flow_from_directory( directory=spiral_params['validation_data_dir'], target_size=(256, 256),
                                                   batch_size=24, class_mode='binary' )

    return (Train_Datagen, Val_Datagen, train_gen, val_gen)


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
        'epochs': 10,
        'batch_size': 24,
        'zoom_range': 0.1,
        'rotation_range': 360,
        'horizontal_flip': False,
        'weights': 'src/spiral.h5'
    }


    model_params = spiral_params

    model = VGG_model()

    train_datagen, validation_datagen, train_generator, validation_generator = data_augmentation()

    # model.load_weights(model_params['weights'])

    model.summary()


    history = model.fit(
        train_generator,
        steps_per_epoch=3,
        epochs=model_params['epochs'],
        validation_data=validation_generator,
        validation_steps=1,
    )

    model.evaluate( validation_generator )

    plot_roc() # Change validation shuffle to False if using plot_roc

    model_evaluation_plot()

    model.save_weights( 'vgg16_spiral.h5' )
