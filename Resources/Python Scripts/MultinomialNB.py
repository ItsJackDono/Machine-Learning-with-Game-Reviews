# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:38:32 2019

@author: itsmr
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score


from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
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
      #a while loop to get the frequncy of positive feautres     
    counter = 0
    while counter < len(feature_names) and input_csv == 'C:/Users/itsmr/Desktop/CS 3rd Year/Semester 1/Intelligent Systems 2019/week 7/ML_Challenge_datasets (CSV)/amazon/reviews_Video_Games_test.csv':
        inst = 4
        
           
        if (instances[inst][counter]) != 0 :
            print ('Positive F Count: ',feature_names[counter], '\t\t Frequency: ', instances[inst][counter])
        
        counter += 1
        
        if counter == len(feature_names):
            print('\n', 'Classfication label for instance 1 is ',labels[inst])
        
    
    return feature_names, instances, labels


######################################
#################################
##invoke the function
if __name__ == '__main__':
#the use of training and testing the classifier 
    training_feature_names, training_instances, training_labels = load_data_from_csv('C:/Users/itsmr/Desktop/CS 3rd Year/Semester 1/Intelligent Systems 2019/week 7/ML_Challenge_datasets (CSV)/amazon/reviews_Video_Games_training.csv')
    test_feature_names, test_instances, test_labels = load_data_from_csv('C:/Users/itsmr/Desktop/CS 3rd Year/Semester 1/Intelligent Systems 2019/week 7/ML_Challenge_datasets (CSV)/amazon/reviews_Video_Games_test.csv')
    #using mulitnomialNB for the single classifier
    classifier = MultinomialNB()
    #classifier = BernoulliNB()
    #classifier = GaussianNB()
    classifier.fit(training_instances, training_labels)
    predicted_test_labels = classifier.predict(test_instances)
    print(classification_report(test_labels, predicted_test_labels, digits=3))
    #q = f1_score(test_labels, predicted_test_labels, average = 'micro')
    #print(q)
    

  