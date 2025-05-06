import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import joblib


df_customer = pd.read_csv('./dataset/Customer.csv')
df_investment = pd.read_csv('./dataset/Funding_and_Investment.csv')
df_SCM = pd.read_csv('./dataset/Supply_Chain.csv')

class KNNClassification:
    def __init__(self, k=3):

        self.vectorizer = TfidfVectorizer(
            max_features=3000,  
            stop_words='english',  
            max_df=0.95,  
            min_df=2 
            )
        self.knn = KNeighborsClassifier(
                n_neighbors=k,
            )
        self.customer_classifer = None
        self.investment_classifer = None
        self.SCM_classifer = None
    def train(self, X, y, classifer_type):
        
        vec = self.vectorizer.fit(X)
        embedding_train = vec.transform(X)
        classifer = self.knn.fit(embedding_train, y)
        if classifer_type == 0:
            self.SCM_classifer = classifer
            joblib.dump(classifer, "./model/SCM_classifer.pkl")
            joblib.dump(vec, "./model/SCM_vec.pkl")
        if classifer_type == 1:
            self.investment_classifer = classifer
            joblib.dump(classifer, "./model/investment_classifer.pkl")
            joblib.dump(vec, "./model/investment_vec.pkl")
        if classifer_type == 2:
            self.customer_classifer = classifer
            joblib.dump(classifer, "./model/customer_classifer.pkl")
            joblib.dump(vec, "./model/customer_vec.pkl")
        
        
    def data_prepare(self, df_small, df_big):
        list_all = []
        list_all.append([])
        list_all.append([])
        list_all.append([])
        titles = df_small['Titles'].fillna('').values.tolist()
        contents = df_big['Summary'].fillna('').values.tolist()
        titles2 = df_big['Title'].fillna('').values.tolist()
        dic = {}
        
        for i in range(len(titles2)):
            dic[titles2[i]] = contents[i]
            
        for i in range(len(titles)):
            title_list = titles[i].split(",")
            print("****************")
            for tmp in title_list:
                list_all[i].append(dic[tmp.strip()])
        X = list_all[0] + list_all[1] + list_all[2]
        y = [0] * len(list_all[0]) + [1] * len(list_all[1]) + [2] * len(list_all[2])
        return X, y
    def train_all(self, df_cluster, big_type):
        print(big_type)
        if big_type == 0:
            X, y = self.data_prepare(df_cluster, df_SCM)
        if big_type == 1:
            X, y = self.data_prepare(df_cluster, df_investment)
        if big_type == 2:
            X, y = self.data_prepare(df_cluster, df_customer)
        self.train(X, y, big_type)

    def predict(self, input_question, big_type):
        if big_type == 0:
            with open('./model/SCM_classifer.pkl', 'rb') as file:
                classifer = joblib.load(file)
            with open('./model/SCM_vec.pkl', 'rb') as file:
                vec = joblib.load(file)
        if big_type == 1:
            with open('./model/investment_classifer.pkl', 'rb') as file:
                classifer = joblib.load(file)
            with open('./model/investment_vec.pkl', 'rb') as file:
                vec = joblib.load(file)           
        if big_type == 2:
            with open('./model/customer_classifer.pkl', 'rb') as file:
                classifer = joblib.load(file)
            with open('./model/customer_vec.pkl', 'rb') as file:
                vec = joblib.load(file)   
        input_question_embedding = vec.transform(input_question)
        
        res = classifer.predict(input_question_embedding)
        
        return res
    