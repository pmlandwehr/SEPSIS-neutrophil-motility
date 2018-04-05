from glob import glob

def main():
    for f in glob("BURNS/*Patient*.xls"):
        if 'Patient' not in f:
            continue
        
        target = "BURNS_CSV/"+f.split("/")[-1] + "_%s.csv"
        f = f.replace(" ", "\\ ")
        target = target.replace(" ", "\\ ")
        cmd = "ssconvert %s %s -S"%(f, target)
        print cmd

if __name__ == "__main__":
    main()
    
