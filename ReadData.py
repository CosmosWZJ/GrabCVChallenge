import scipy.io
import numpy as np
import pandas as pd
import PIL

def nparray_map(f, array) -> np.array:
    return np.array(list(map(f, array)))

def read_train_data():
    #returns a list of training photos' paths and labels
    mat = scipy.io.loadmat('./devkit/cars_train_annos')
    table = mat['annotations']
    fileName = nparray_map(lambda  _: _[0], table['fname'][0])
    classes = nparray_map(lambda  _: _[0][0], table['class'][0])

    pdata = pd.DataFrame(columns=['fileName','class'])
    pdata['fileName'] = pd.Series(fileName)
    pdata['class'] = pd.Series(classes)
    pdata['class'] = pdata['class'].apply(str)
    return pdata

def read_test_x():
    mat = scipy.io.loadmat('./devkit/cars_test_annos')
    table = mat['annotations']
    fileName = nparray_map(lambda _: _[0], table['fname'][0])
    pdata = pd.DataFrame(columns=['fileName'])
    pdata['fileName'] = pd.Series(fileName)
    #print(pdata)
    return pdata['fileName']

def read_test_y():
    data = pd.read_csv('./devkit/train_perfect_preds.txt', header=None)
    pdata = pd.DataFrame(data=data)
    # pdata['class'] = pd.Series(data)
    #print(pdata[0])
    return pdata;

def read_test_data():
    x_valid = read_test_x()
    y_valid = read_test_y()
    pdata = pd.DataFrame(columns=['fileName', 'class'])
    pdata['fileName'] = pd.Series(x_valid)
    pdata['class'] = y_valid
    pdata['class'] = pdata['class'].apply(str)
    #print(pdata)
    return pdata

#read_test_data()

# train_frame = read_train_data()
# print(train_frame)
#
# test_frame = train_frame.iloc[int(8144*0.8):]
# train_frame = train_frame.iloc[:int(8144*0.8)]
#
# print(test_frame)
# print(train_frame)