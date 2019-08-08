import os
from glob import glob
import pandas as pd

BASE_DIR = "../data"

def concat_words(row):
    return row['word'].replace(" ", "_" ).lower()

fns = glob(os.path.join(BASE_DIR, 'train_simplified', '*.csv.gz'))
dfts = []
for fn in fns:
    print(fn)
    dft = pd.read_csv(fn, nrows=500)
    dft['idx'] = range(500)
    dft['word'] = dft.apply(concat_words, axis=1)
    dfts.append(dft)

dft = pd.concat(dfts)
dft.to_csv("../temp/drawings_500perclass.csv")
