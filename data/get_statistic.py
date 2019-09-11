# get the ratio of classes
import pandas as pd
from collections import  Counter
import math
import numpy as np
ratio = 0.2
def getTrainAndValIndex():
    table = pd.read_csv("../dataset/Train_label.csv", header=None)
    csv_narray = table.values  #[6392, 2]
    classes_num = csv_narray[1:, 1]
    classes_statistic = Counter(classes_num)


    train_datasets = []
    val_datasets=[]

    val_statistic = {}
    for class_id, class_num in classes_statistic.items():
        val_statistic[class_id] = math.floor(class_num * ratio)

    val_statistic_now = {}
    for class_id, class_num in classes_statistic.items():
        val_statistic_now[class_id] = 0
    for element in csv_narray[1:,]:
        if (val_statistic_now[element[1]] >= val_statistic[element[1]]):
            train_datasets.append(element)
        else:
            val_datasets.append(element)
            val_statistic_now[element[1]] = val_statistic_now[element[1]] + 1
    return np.array(train_datasets), np.array(val_datasets)

if __name__ == "__main__":
    tain, val = getTrainAndValIndex()
    i = 0