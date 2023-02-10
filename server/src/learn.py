import os

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import random
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxconverter_common import add_metadata_props
import constants
from datetime import date


def state_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def get_hash():
    os.system('git rev-parse HEAD > hash.txt')
    with open('hash.txt', 'r') as f:
        hash = f.readline().strip()
    os.remove('hash.txt')
    return hash


def create_model(args):
    df = pd.read_csv(args.infile)

    X = df.drop(constants.FIELD_IS_TEXT, axis=1).values
    y = df[constants.FIELD_IS_TEXT].values

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=args.seed)

    parameters = {'max_depth': [3, 4, 5], 'min_samples_split': range(2, 10)}
    rf = RandomForestClassifier(random_state=args.seed)
    clf = GridSearchCV(rf, parameters, cv=3)
    clf.fit(X_train, y_train)

    estimator = clf.best_estimator_
    print(f'Best score on CV {clf.best_score_:.3},'
          f'best params {clf.best_params_}')
    print(f'Accuracy on validation set '
          f'{accuracy_score(estimator.predict(X_val), y_val):.3}')

    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(estimator, initial_types=initial_type)
    hash = get_hash()

    params = {'hash_commit': str(hash),
              'created': date.today().__str__(),
              'experiment': args.experiment}
    add_metadata_props(onx, params, 15)

    with open(args.outfile, 'wb') as f:
        f.write(onx.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Makes ML model in .onnx format')
    parser.add_argument('--infile',
                        type=str,
                        required=True,
                        help='Input .csv dataframe for learning')
    parser.add_argument('--outfile',
                        type=str,
                        required=True,
                        help='Output, .onnx file for pretrained classification'
                             'model')
    parser.add_argument('--experiment',
                        type=str,
                        required=True,
                        help='Experiment name')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for model')

    args = parser.parse_args()
    create_model(args)
