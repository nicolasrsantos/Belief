#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:59:30 2018

@author: nicolas
"""
import numpy as np
import pandas as pd
import os

def getBiggestValueIndex(list):
    if all(v == 0 for v in list):
        return -1
    elif list[0] == list[1] and list[2] < list[1]:
        return -1
    elif list[0] == list[2] and list[1] < list[2]:
        return -1
    elif list[1] == list[2] and list[0] < list[1]:
        return -1
    
    return max(range(len(list)), key = list.__getitem__)
    
    
if __name__ == "__main__":
    csv_path = "/home/nicolas/Documents/Scripts/Rumour Belief/csv/"
    csv_list = []
    for filename in os.listdir(csv_path):
        csv_list.append(pd.read_csv(csv_path + filename))
    
    # index 0 = Believe; index 1 = Neutral; index 2 = Disbelieve
    class_per_submission = [[0, 0, 0] for i in range (0, 541)]
    
    for csv in csv_list:
        for i in range (0, 541):
            if csv.Label[i] == 'Believe':
                class_per_submission[i][0] += 1
            elif csv.Label[i] == 'Neutral':
                class_per_submission[i][1] += 1
            elif csv.Label[i] == 'Disbelieve':
                class_per_submission[i][2] += 1
                
    for i in range (0, 541):
        max_index = getBiggestValueIndex(class_per_submission[i])
        if max_index == 0:
            print("Believe")
        elif max_index == 1:
            print("Neutral")
        elif max_index == 2:
            print("Disbelieve")
        elif max_index == -1:
            print("Remove this row")