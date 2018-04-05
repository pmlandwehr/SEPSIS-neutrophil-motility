from glob import glob
import os
from tqdm import trange

import numpy as np
import numpy.random as npr
import pandas as pd
from matplotlib import pylab as plt


def hex_to_rgb(value):
    lv = len(value)
    return tuple(
        int(value[i:i + lv // 3], 16) / 255. for i in range(0, lv, lv // 3)
    )

def cell_type_from_file(f):
    print(f)
    if "PRESE" in f:
        return 0
    if "t02" in f.lower():
        return 1
    if "t04" in f.lower():
        return 1
    return 0

category10_in_hex = [
      "1f77b4",
      "ff7f0e",
      "2ca02c",
      "d62728",
      "9467bd",
      "8c564b",
      "e377c2",
      "7f7f7f",
      "bcbd22",
      "17becf"]
category10 = [hex_to_rgb(val) for val in category10_in_hex]

files = npr.permutation(glob(os.path.join('csvs', '*.csv')))

def files_to_cells(files):
    cell_accum = []
    cell_status = []
    for f in files:
        try:
            frame = pd.read_csv(f)
        except Exception as e:
            print('===')
            print('{} <<Failed to read'.format(f))
            print(e)
            print('===')
            continue
        track_id = frame['TRACK_ID']
        pos_x = frame['POSITION_X']
        pos_y = frame['POSITION_Y']
        t = frame['POSITION_T']
        estimated_diameter = frame['ESTIMATED_DIAMETER']
        cell_type = cell_type_from_file(f)
        if cell_type is None:
            continue

        cells = []
        # separate out by cell
        uniques = np.unique(track_id)
        #print uniques

        for n in uniques:
            sx = pos_x[track_id == n]
            sy = pos_y[track_id == n]
            st = t[track_id == n]
            sest_d = estimated_diameter[track_id == n]
            ssnr = snr[track_id == n]
            scontrast = contrast[track_id == n]
            cell = sorted(zip(st.values, sx.values, sy.values, sest_d.values, ssnr.values, scontrast.values), key=lambda x: x[0])
           #cell = sorted(zip(st.values, sx.values, sy.values, sest_d.values), key=lambda x: x[0])
            cell = np.asarray(cell)
            cells.append(cell)

            cell_accum.append(cell)
            cell_status.append(cell_type)
    return cell_accum, cell_status


def features(paths, label):
    # positions is list of t, x, y, est_diam, snr, contrast
    t = paths[:, 0]
    x = paths[:, 1]
    y = paths[:, 2]
    d = paths[:, 3]
    snr = paths[:, 4]
    contrast = paths[:, 5]

    dx = np.abs((x[1:] - x[0:-1]) / np.diff(t))
    dy = np.abs((y[1:] - y[0:-1]) / np.diff(t))
    #plt.plot(dx, dy, '-o')
    #plt.show()

    ddx = np.diff(x)
    ddy = np.diff(y)

    return len(y), np.var(ddx), np.var(np.sign(dx))

dd = {}
dd["Healthy"] = 0
dd["T04"] = 1
dd["T11"] = 2
dd["T02"] = 3
dd["T03"] = 4
dd["T08"] = 5
dd["UNKNOWN"] = 6

idd = {v:k for k,v in dd.items()}


plts = [["Healthy"], ["T04", "T02"], ["T11", "T03", "T08"]]

def do_tsne():
    cell_accum, cell_status = files_to_cells(files)
    feats_accum = []
    for i, (x, status) in enumerate(zip(cell_accum, cell_status)):
        feats_accum.append(features(x))

    feats_accum = np.asarray(feats_accum)
    cell_status = np.asarray(cell_status)

    from tsne import bh_sne
    points = bh_sne(feats_accum.astype("float64"))

    for c, p in zip(category10, plts):
        mask = np.zeros_like(points[:, 0])
        print p
        for pi in p:
            print pi, dd[pi]
            mask[cell_status == dd[pi]] = 1
        print np.sum(mask)
        mask = (mask==1)
        plt.plot(points[:, 0][mask], points[:, 1][mask], "o", c=c, lw=0, alpha=0.5)

    plt.legend(["Healthy", "Septic", "Non-Septic"])
    plt.show()
S

def features_for_files(files):
    cell_accum, cell_status = files_to_cells(files)
    feats_accum = []
    for i, (x, status) in enumerate(zip(cell_accum, cell_status)):
        feats_accum.append(features(x, status))

    feats_accum = np.asarray(feats_accum)
    cell_status = np.asarray(cell_status)

    plts = [["T04", "T02"], ["T11", "T03", "T08"]]
    Y = [i for i in cell_status]
    X = [x for x,i in zip(feats_accum, cell_status)]
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
lr = LogisticRegression()

np.random.shuffle(files)
all_data = [features_for_files([f]) for f in files]
all_data, files = zip(*[(a,f) for (f,a) in zip(files, all_data) if len(a[0]) != 0])
print len(all_data), "Valid files found"
all_X, all_Y = zip(*all_data)
all_X = np.asarray(all_X)
all_Y = np.asarray(all_Y)

def pat_id(f):
    tXX = f.split("-")[1]
    ds = f.split("-")[2]
    r = tXX+"_"+ds
    print r
    return r

patient_id = []
for f in files:
    patient_id.append(pat_id(f))

data = []
for y in all_Y:
    data.append(np.mean(y))

print np.mean(data)
from collections import defaultdict

def run_once():
    # train test split on a patient basis
    pos_pat = defaultdict(lambda: [])
    neg_pat = defaultdict(lambda: [])
    all_pat = defaultdict(lambda: [])
    for i, (pat_id, X, Y) in enumerate(zip(patient_id, all_X, all_Y)):
        if np.mean(Y) == 1.0:
            pos_pat[pat_id].append((i, X))
        else:
            neg_pat[pat_id].append((i, X))
        all_pat[pat_id].append((i, X, Y))

    # TODO probabilistically round up or down
    n_neg = int(len(neg_pat.keys())*.5)
    # 5 pos, 7 neg
    trPatIds = np.random.choice(pos_pat.keys(), size=2, replace=False).tolist()\
               + np.random.choice(neg_pat.keys(), size=2, replace=False).tolist()
    tePatIds = list(set(patient_id) - set(trPatIds))

    idxs = []

    trFX = []
    trFY = []
    teFX = []
    teFY = []
    for p in trPatIds:
        for pp in all_pat[p]:
            index, X, Y = pp
            trFX.append(X)
            trFY.append(Y)

    for p in tePatIds:
        for pp in all_pat[p]:
            index, X, Y = pp
            teFX.append(X)
            teFY.append(Y)

    X = np.concatenate(trFX, axis=0)
    Y = np.concatenate(trFY, axis=0)

    idxs = np.arange(len(Y))
    np.random.shuffle(idxs)
    spt = int(len(X)*.9)

    trX = X[idxs[0:spt]]
    trY = Y[idxs[0:spt]]
    teX = X[idxs[spt:]]
    teY = Y[idxs[spt:]]


    lr.fit(trX, trY)
    p = lr.predict_proba(trX)
    fpr, tpr, _ = roc_curve(trY, p[:, 1])

    p = lr.predict_proba(teX)
    fpr, tpr, _ = roc_curve(teY, p[:, 1])

    # Make a new X,Y for heldout
    X = np.concatenate(teFX, axis=0)
    Y = np.concatenate(teFY, axis=0)
    p = lr.predict_proba(X)
    fpr, tpr, _ = roc_curve(Y, p[:, 1])
    plt.plot(fpr, tpr)
    heldout_auc = roc_auc_score(Y, p[:, 1])

    preds = []
    actual = []
    for i in range(len(teFY)):
        X = np.concatenate(teFX[i:i+1], axis=0)
        Y = np.concatenate(teFY[i:i+1], axis=0)

        if len(X) <= 0:
            #print "no data", files[i]
            continue

        p = lr.predict_proba(X)
        preds.append(np.mean(p[:, 1]))
        actual.append(np.mean(Y))

    heldout_accur = accuracy_score(np.asarray(actual).astype("int"), np.asarray(preds) >= 0.5)
    print heldout_accur
    print "Of the %d files looked at"%len(preds)
    return heldout_auc

accurs = [run_once() for x in trange(50)]
print np.mean(accurs), "mean_auc"
plt.show()
