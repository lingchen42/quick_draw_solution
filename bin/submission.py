import sys
import pandas as pd
from keras.models import load_model
from data_gen import *
from helper import *

INPUT_DIR = "../data"
size = 128

def f2cat(filename: str) -> str:
    return filename.split('.')[0]


def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def test_predict(model):
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    x_test = df_to_image_array_xd(test, size)

    print(test.shape, x_test.shape)
    print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))

    test_predictions = model.predict(x_test, batch_size=128, verbose=1)
    top3 = preds2catids(test_predictions)

    cats = list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)

    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    
    print("Estimating map3...")
    valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
    
    print("Writing to gs_mn_submission_{}.csv".format(int(map3 * 10**4)))
    submission.to_csv('../submission/qd_submission_{}.csv'.format(int(map3 * 10**4)), index=False)

if __name__ = "__main__":
    # load model
    print("Load model ...")
    model = load_model(sys.argv[1]) 
    test_predict(model)