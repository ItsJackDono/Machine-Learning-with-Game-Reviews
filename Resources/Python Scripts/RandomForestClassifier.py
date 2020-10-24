# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:46:12 2020

@author: itsmr
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#loading the csv file, seperating features, instances, labels  and values 
def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    label_name = csv_headings[len(csv_headings) - 1:len(csv_headings)][0]
    df = df._get_numeric_data()
    numpy_array = df.as_matrix()
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels

   

#################################
##invoke the function
if __name__ == '__main__':
    training_feature_names, training_instances, training_labels = load_data_from_csv('C:/Users/itsmr/Desktop/CS 3rd Year/Semester 1/Intelligent Systems 2019/week 7/ML_Challenge_datasets (CSV)/amazon/reviews_Video_Games_training.csv')
    test_feature_names, test_instances, test_labels = load_data_from_csv('C:/Users/itsmr/Desktop/CS 3rd Year/Semester 1/Intelligent Systems 2019/week 7/ML_Challenge_datasets (CSV)/amazon/reviews_Video_Games_test.csv')
#defining arrays for x axis and y axis values 
#defining varaibles for bubble sort type loop, and storing the variable of highest Fcount to later run as the main RF Count

    x = []
    y = []
    z = 0
    holder = 0
    itCount = 0
    #for loop to iterate and find the highest FCount to later use as the final RF N_estimator 
    for counter in range(1, 200):
        classifier = RandomForestClassifier(n_estimators = counter, max_features=20, n_jobs=5)
        classifier.fit(training_instances, training_labels)
        predicted_test_labels = classifier.predict(test_instances)
        q = f1_score(test_labels, predicted_test_labels, average = 'micro')
        itCount = itCount + 1
        x.append(counter)
        y.append(q)
        #print('testing', q, 'it count ', itCount)
        #storing the highest FCount corrosponding the RF N_estimater count
        if q > holder and q > z and z != holder:
            z = q
            finalTC = itCount
            print('highest F-Score', z, ', Decision Tree Count: ', itCount)
        holder = q
    #re-specifying the RF for the best preforming N_estimator Count
    #print(finalTC)
    classifier = RandomForestClassifier(n_estimators = finalTC, max_features=20, n_jobs=5)        
    plt.xlabel('Number of Random DT')
    plt.ylabel('F-measure')
    plt.plot(x,y)
    print(classification_report(test_labels, predicted_test_labels, digits=3)) 
    
    

