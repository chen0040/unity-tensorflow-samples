import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
import os


def main():

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'

    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    Xtest = Xtest.astype('float32') / 255

    with tf.gfile.FastGFile(current_dir + '/models/tf/cnn_cifar10.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        [print(n.name) for n in sess.graph.as_graph_def().node]
        predict_op = sess.graph.get_tensor_by_name('output_node0:0')

        for i in range(Xtest.shape[0]):
            x = Xtest[i]
            x = np.expand_dims(x, axis=0)
            y = Ytest[i]
            predicted = sess.run(predict_op, feed_dict={"conv2d_1_input:0": x,
                                                        'dropout_1/keras_learning_phase:0': 0})

            predicted_y = np.argmax(predicted, axis=1)
            print('actual: ', y, '\tpredicted: ', predicted_y)


if __name__ == '__main__':
    main()
