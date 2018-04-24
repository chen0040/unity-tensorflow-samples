import sys
import os


def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    from cifar10.cifar10_classifier import Cifar10Classifier

    classifier = Cifar10Classifier()
    classifier.load_model(current_dir + '/models')
    classifier.export_tensorflow_model(output_fld=os.path.join(current_dir, 'models', 'tf'))


if __name__ == '__main__':
    main()

