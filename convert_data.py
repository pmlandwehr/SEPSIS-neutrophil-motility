from glob import glob
import os

def main():
    for f in glob(os.path.join('TRACKS4', 'Spots*.xls')):
        target = os.path.join('csvs', '{}.csv'.format(f.split(os.path.sep)[-1]))
        f = f.replace(' ', '\\ ')
        if '(' in f:
            continue
        target = target.replace(' ', '\\ ')
        cmd = 'ssconvert {} {}'.format(f, target)
        print(cmd)
   
if __name__ == "__main__":
    main()
    
