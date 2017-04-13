
import numpy as np

def build_input_train(data_path):
    result = []
    with open(data_path) as train_file:
        for line in train_file:
            lineContent = line.split(',')
            if lineContent[0] == 'label':
                continue
            tag = int(lineContent[0])
            img = np.array([int(i) for i in lineContent[1:]])
            result.append((tag,img))
    return result

def build_input_eval(data_path):
    result = []
    with open(data_path) as train_file:
        for line in train_file:
            lineContent = line.split(',')
            if lineContent[0] == 'pixel0':
                continue
            img = np.array([int(i) for i in lineContent])
            tag = 0
            result.append((tag,img))
    return result