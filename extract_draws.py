# -*- coding: utf-8 -*-
import os
from glob import glob
import pandas as pd
from six import StringIO


def process_draw_file(f):
    content = '\n'.join(open(f).readlines()[1:])
    ff = StringIO(content)

    frame = pd.read_csv(ff)
    try:
        del frame['MATLAB']
        del frame['MATLAB.1']
        del frame['MATLAB.2']
    except:
        pass
    # control 0-8
    # fMLP 9-18
    # ltb4 18-
    ss = [u'Track n°', u'Slice n°', u'X', u'Y', u'Distance', u'Velocity', u'Pixel Value']
    ss = [x.encode("utf-8") for x in ss]
    #control = frame[frame.columns[0:8]].dropna()
    #fmlp = frame[frame.columns[9:17]].dropna()
    #ltb4 = frame[frame.columns[18:]].dropna()
    control = frame[ss].dropna()
    fmlp = frame[[s+".1" for s in ss]].dropna()
    ltb4 = frame[[s+".2" for s in ss]].dropna()

    column_names = ['track_n', 'slice_n', 'x', 'y', 'distance', 'velocity', 'pixel_value']

    patient_string = "_".join(f.split("/")[1].split("_")[1:3]).split(".")[0]
    draw_num = f.split("Draw")[1].split("-V")[0]

    path = os.path.join('converted_data', patient_string, 'control_{}.csv'.format(draw_num))
    
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

    control.to_csv(path, header=column_names)
    
    path = os.path.join('converted_data', patient_string, 'fmlp_{}.csv'.format(draw_num))    
    fmlp.to_csv(path, header=column_names)
    
    path = os.path.join('converted_data', patient_string, 'ltb4_{}.csv'.format(draw_num))   
    ltb4.to_csv(path, header=column_names)

    
def main():
    files = glob("/Patient_*Draw*-V*.csv")
    for i, f in enumerate(files):
        print('{} {}'.format(i, f))
        process_draw_file(f)
 

if __name__ == "__main__":
    main()
