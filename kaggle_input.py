
import numpy as np

def build_input(data_path):
    result = []
    with open(data_path) as train_file:
        for line in train_file:
            lineContent = line.split(',')
            if lineContent[0] == 'label':
                continue
            tag = int(lineContent[0])
            img = np.array([int(i) for i in lineContent[1:]]).reshape((28,28,1))
            result.append((tag,img))
    return result
