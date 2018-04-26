from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
import tensorflow as tf
import keras.backend as K


class Cifar10Classifier:
    model_name = 'cnn_cifar10'

    def __init__(self):
        self.model = None
        self.input_shape = None
        self.nb_classes = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10Classifier.model_name + '_architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10Classifier.model_name + '_weights.h5')

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10Classifier.model_name + '_config.npy')

    def load_model(self, model_dir_path):

        config_file_path = self.get_config_file_path(model_dir_path)
        weight_file_path = self.get_weight_file_path(model_dir_path)

        if not os.path.exists(config_file_path):
            return
        if not os.path.exists(weight_file_path):
            return
        
        config = np.load(config_file_path).item()

        self.input_shape = config['input_shape']
        self.nb_classes = config['nb_classes']

        self.model = self.create_model(self.input_shape, self.nb_classes)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.load_weights(weight_file_path)

    def predict_label(self, filename):
        img = Image.open(filename)

        if K.image_data_format() == 'channels_first':
            _, img_width, img_height = self.input_shape
        else:
            img_width, img_height, _ = self.input_shape

        img = img.resize((img_width, img_height), Image.ANTIALIAS)

        input = np.asarray(img)
        input = input.astype('float32') / 255
        input = np.expand_dims(input, axis=0)

        print(input.shape)

        predicted_class = self.model.predict_classes(input)[0]

        labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
        return predicted_class, labels[predicted_class]

    @staticmethod
    def create_model(input_shape, nb_classes):
        model = Sequential()
        model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(rate=0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(rate=0.25))

        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=nb_classes))
        model.add(Activation('softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def run_test(self, current_dir):
        print(self.predict_label(current_dir + '/../training/bi_classifier_data/training/cat/cat.2.jpg'))

    def fit(self, Xtrain, Ytrain, model_dir_path, input_shape=None, nb_classes=None, test_size=None, batch_size=None,
            epochs=None):

        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        if test_size is None:
            test_size = 0.2

        if input_shape is None:
            input_shape = (32, 32, 3)

        if nb_classes is None:
            nb_classes = 10

        Xtrain = Xtrain.astype('float32') / 255
        Ytrain = np_utils.to_categorical(Ytrain, nb_classes)

        self.input_shape = input_shape
        self.nb_classes = nb_classes

        config_file_path = self.get_config_file_path(model_dir_path)

        config = dict()
        config['input_shape'] = input_shape
        config['nb_classes'] = nb_classes

        np.save(config_file_path, config)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        self.model = self.create_model(input_shape, nb_classes)

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = self.model.fit(x=Xtrain, y=Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_split=test_size,
                                 callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        np.save(os.path.join(model_dir_path, Cifar10Classifier.model_name + '-history.npy'), history.history)

        return history

    def evaluate(self, Xtest, Ytest, batch_size=None):

        if batch_size is None:
            batch_size = 64

        Xtest = Xtest.astype('float32') / 255
        Ytest = np_utils.to_categorical(Ytest, self.nb_classes)

        return self.model.evaluate(x=Xtest, y=Ytest, batch_size=batch_size, verbose=1)

    def export_tensorflow_model(self, output_fld, output_model_file=None,
                                output_graphdef_file=None,
                                num_output=None,
                                quantize=False,
                                save_output_graphdef_file=False,
                                output_node_prefix=None):

        K.set_learning_phase(0)

        if output_model_file is None:
            output_model_file = Cifar10Classifier.model_name + '.pb'

        if output_graphdef_file is None:
            output_graphdef_file = 'model.ascii'
        if num_output is None:
            num_output = 1
        if output_node_prefix is None:
            output_node_prefix = 'output_node'

        pred = [None] * num_output
        pred_node_names = [None] * num_output
        for i in range(num_output):
            pred_node_names[i] = output_node_prefix + str(i)
            pred[i] = tf.identity(self.model.outputs[i], name=pred_node_names[i])
        print('output nodes names are: ', pred_node_names)

        sess = K.get_session()

        if save_output_graphdef_file:
            tf.train.write_graph(sess.graph.as_graph_def(), output_fld, output_graphdef_file, as_text=True)
            print('saved the graph definition in ascii format at: ', output_graphdef_file)

        from tensorflow.python.framework import graph_util
        from tensorflow.python.framework import graph_io
        # from tensorflow.tools.graph_transforms import TransformGraph
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
        print('saved the freezed graph (ready for inference) at: ', output_model_file)
