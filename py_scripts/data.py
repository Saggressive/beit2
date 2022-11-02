import os
import csv

if __name__=="__main__":
    data=[]
    with open("ir_data/raw.csv") as f:
        f_csv=csv.reader(f)
        header=next(f_csv)
        for row in f_csv:
            data.append(row[0])
    data=list(set(data))
    data=[i+"\n" for i in data]
    with open("ir_data/paser.txt","w") as f:
        f.writelines(data)
