import glob
import os
from collections import defaultdict
from tqdm import trange

import numpy as np
import numpy.random as npr
import pandas as pd
from matplotlib import pylab as plt

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_raw(patient_name, draw_num):
    """Read in a pandas.DataFrame from converted_data/<patient_name>/control_<draw_num>.csv"""
    path = os.path.join('converted_data', patient_name, 'control_{}.csv'.format(draw_num))
    if not os.path.exists(path):
        print('{} Not found'.format(path))
        return None
    frame = pd.read_csv(path)
    return frame

def features(frame, draw_num):
    features = []
    has_cells = (len(frame['x']) != 0)
    features.append(has_cells)
    # Maybe remove, possibly data leakage....
    features.append(draw_num)
    if has_cells == False:
        return features + [0]*8
        return features + [0]*17

    num_cells = np.max(frame['track_n'])
    features.append(num_cells > 10)
    features.append(num_cells > 20)
    features.append(num_cells)

    #Parameters: N, O, P, R, MD, AD, V, A, Delta V, Delta A

    dx = np.diff(frame['x'])
    ddx = np.diff(dx)
    features.append(np.mean(dx))
    features.append(np.mean(np.abs(dx)))
    features.append(np.var(dx))
    features.append(np.sum(np.abs(np.sign(ddx))))

    dy = np.diff(frame['y'])
    ddy = np.diff(dy)
    features.append(np.mean(dy))
    features.append(np.mean(np.abs(dy)))
    features.append(np.var(dy))
    features.append(np.sum(np.abs(np.sign(ddy))))

    features.append(np.mean(frame['velocity']))
    features.append(np.var(frame['velocity']))

    pixel_value = frame['pixel_value']
    features.append(np.mean(pixel_value))
    features.append(np.var(pixel_value))

    distance = frame['distance']
    features.append(np.mean(distance))
    features.append(np.var(distance))

    return features

def load_labels():
    """Read in a file of labels with each row formatted as <name>:<draw>[,<draw>][,<draw>]... and save as a dict keyed on name"""
    label_dict = {}
    for line in open(os.path.join('converted_data', 'labels.txt'), 'r'):
        name, draws = line.strip().split(":")
        print(draws)
        draws = [int(x) for x in draws.split(",") if x != '']
        label_dict[name] = draws
    return label_dict

def gather_feats_labels(patients, name_to_feats, name_to_labels):
    X = []
    Y = []
    for p in patients:
        X.extend(name_to_feats[p])
        Y.extend(name_to_labels[p])
    X,Y = zip(*[(x,y) for x,y in zip(X, Y) if x is not None])
    return np.asarray(X), np.asarray(Y)

def shuffle_train(X, Y, teX):
    """Fit a model on shuffled samples and labels X and Y, and return prediction on the training data and on test samples TeX"""    
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.7))

    idx = npr.permutation(np.arange(len(X)))
    pipe.fit(X[idx], Y[idx])
    
    return pipe.predict_proba(X)[:, 1], pipe.predict_proba(teX)[:, 1]

def main():
    name_to_labels = load_labels()
    for name in name_to_labels:
        name_to_labels[name] = [0 if i == 0 else 1 for i in name_to_labels[name]]
    
    name_to_feats = defaultdict(list)
    for name in name_to_labels:
        for i in range(len(name_to_labels[name])):
            frame = load_raw(name, i+1)
            if frame is None:
                name_to_feats[name].append(None)
            else:
                name_to_feats[name].append(features(frame, draw_num=i))

    # max_feats = max([len(x) for x in name_to_feats.values()[0] if x is not None])
    # for k,v in name_to_feats.items():
        # for i in range(len(v)):
            #v[i] = v[i] + [0]*(max_feats - len(v[i]))


    # fix the features where can't make with zeros

    accum = []
    for x in trange(10000):
        patients = name_to_labels.keys()
        n_train = int(len(patients) * .8)
        train_pat = npr.choice(patients, size=n_train).tolist()
        test_pat = list(set(patients) - set(train_pat))

        train_X, train_Y = gather_feats_labels(train_pat, name_to_feats, name_to_labels)
        test_X, test_Y = gather_feats_labels(test_pat, name_to_feats, name_to_labels)
        
        if len(np.unique(test_Y)) == 1:
            continue
        if len(np.unique(train_Y)) == 1:
            continue
        
        X, Y = gather_feats_labels(patients, name_to_feats, name_to_labels)
        idx = npr.permutation(np.arange(len(X)))
        partition = int(len(X)*.8)
        train_X = X[idx[:partition]]
        test_X = X[idx[partition:]]
        train_Y = Y[idx[:partition]]
        test_Y = Y[idx[partition:]]

        train_Y_pred, test_Y_pred = shuffle_train(train_X, train_Y, test_X)

        heldout_auc = roc_auc_score(test_Y, test_Y_pred)
        train_auc = roc_auc_score(train_Y, train_Y_pred)

        fpr, tpr, _ = roc_curve(test_Y, test_Y_pred)
        accum.append([heldout_auc, train_auc])
    
    print('Mean: {}, Variance: {}'.format(np.mean(accum, axis=0), np.var(accum, axis=0)))
    accum = np.asarray(accum)
    plt.subplot(2,1,1)
    plt.title("Histogram of AUC from ML based model")
    plt.hist(accum[:, 0], bins=100, normed=True, range=[0,1])
    plt.xlabel("Validation AUC")
    plt.subplot(2, 1, 2)
    plt.hist(accum[:, 1], bins=100, normed=True, range=[0,1])
    plt.xlim([0, 1])
    plt.xlabel("Train AUC")
    plt.show()
    
    
if __name__ == "__main__":
    main()
