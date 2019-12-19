from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.externals import joblib
import sys
from getopt import getopt
import os


def train(features_csv, scores_csv, output_dir):
    RF = RandomForestClassifier()
    # FCN output features
    features = pd.read_csv(features_csv)
    # scoring label
    scores = pd.read_csv(scores_csv)
    # all names of files
    files = []
    for i in range(features.size):
        files.append(features.iloc[i, 0])
    # train size 80%
    size = int(0.8*len(files))
    try:
        train_feature = features[files].iloc[:size, 1:]
        train_score = scores[files].iloc[:size, 1]
        test_feature = features[files].iloc[size:, 1:]
        test_score = scores[files].iloc[size:, 1:]
    except Exception:
        print('error when train RF')
        exit(-1)
    # fit
    RF.fit(train_feature, train_score)
    # test score
    print(RF.score(test_feature, test_score))
    # save params
    joblib.dump(RF, output_dir)

def predict(feature_csv, pkl_dir, output_file):
    fields = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']

    RFs = []
    for field in fields:
        RF_path = os.path.join(pkl_dir, field, '.pkl')
        RFs.append(joblib.load(RF_path))

    features = pd.read_csv(feature_csv)
    scores = pd.DataFrame(columns=['id', 'L', 'B', 'beautiful', 'boring',
                                   'depressing', 'lively', 'safety', 'wealthy'])

    for i in range(len(features)):
        feature = features.iloc[i]
        # split file name
        loc = feature[0].split('_')
        # one row
        score = []
        # id
        score.append(i)
        # L
        score.append(loc[0])
        # B
        score.append(loc[1])
        for j in range(len(fields)):
            s = RFs[j].predict(feature[1:])
            score.append(s)
        # add row
        scores.iloc[i] = score

    scores.to_csv(output_file)


if __name__ == '__main__':
    input_dir = './csv_data'
    output_dir = './RF_pkl'
    fields = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']
    if(len(sys.argv) > 1):
        argv = sys.argv[1:]
        opts, args = getopt(argv, 'i:o:', ['input_dir=', 'output_dir='])
        for opt, arg in opts:
            if opt in ['-i', 'input_dir']:
                input_dir = arg
            elif opt in ['-o', 'output_dir']:
                output_dir = arg
    for field in fields:
        features_csv = os.path.join(input_dir, field, '_features.csv')
        scores_csv = os.path.join(input_dir, field, '_scores.csv')
        output_pkl = os.path.join(output_dir, field, '.pkl')
        if os.path.exists(features_csv) and os.path.exists(scores_csv):
            train(features_csv, scores_csv, output_pkl)

