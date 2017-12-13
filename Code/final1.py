import time
st = time.time()
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import sys
sys.path.append('/usr/local/lib/python36/site-packages')
import cv2

def imres(pt):

    img = cv2.imread(pt)
    res = cv2.resize(img, (64, 64), cv2.INTER_LINEAR) 
    return [pt, res]


def nimpath(pt):

    try:
        nim = Image.open(pt)
        return [pt, {'size': nim.size}]
    except:
        print(pt)
        return [pt, {'size': [0,0]}]


def standardize(pt):

    newd = {}
    vl = Pool(cpu_count())
    var = vl.map(imres, pt)

    for i in range(len(var)):
        newd[var[i][0]] = var[i][1]
    var = []

    stndt = [newd[i] for i in pt]
    stndt = np.array(stndt, dtype=np.uint8)
    stndt = stndt.transpose((0, 3, 1, 2))
    stndt = (stndt.astype('float32'))/255
    #stndt = stndt / 255
    return stndt


def preprocessing():

    train = glob.glob('../train/**/*.jpg')
    train = pd.DataFrame([[p.split('/')[2],p.split('/')[3],p] for p in train], columns = ['type','image','path'])[::4] 
    #print(train)

    sdic = {}
    vl = Pool(cpu_count())
    var = vl.map(nimpath, train['path'])
    for i in range(len(var)):
        sdic[var[i][0]] = var[i][1]
    train['size'] = train['path'].map(lambda x: ' '.join(str(s) for s in sdic[x]['size']))

    #train = proc_cnt(train)
    train = train[train['size'] != '0 0'].reset_index(drop=True)

    #print(train)

    dt = standardize(train['path'])

    #print(dt)
    np.save('trainnew.npy', dt, allow_pickle=True, fix_imports=True)

    stm = LabelEncoder()

    ttrans = stm.fit_transform(train['type'].values)

    #print(ttrans)
    np.save('traintrans.npy', ttrans, allow_pickle=True, fix_imports=True)

    test = glob.glob('../test/*.jpg')

    test = pd.DataFrame([[p.split('/')[2],p] for p in test], columns = ['image','path']) #[::25] 
    
    tsd = standardize(test['path'])
    #print(tsd)

    np.save('testnew.npy', tsd, allow_pickle=True, fix_imports=True)

    timname = test.image.values

    #print(timname)
    np.save('test_idnew.npy', timname, allow_pickle=True, fix_imports=True)

if __name__ == '__main__':

    preprocessing()
    end = time.time()
    print(end-st)