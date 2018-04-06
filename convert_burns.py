import os
from glob import glob

def main():
    for f in glob(os.path.join('BURNS', '*Patient*.xls')):       
        
        fname = f.rsplit(os.path.sep, 1)[-1]
        target = os.path.join('BURNS_CSV', '{}_%s.csv'.format(fname))
        
        cmd = 'ssconvert {} {} -S'.format(f.replace(' ', '\\ '), target.replace(' ', '\\ '))
        print(cmd)

if __name__ == "__main__":
    main()
    
