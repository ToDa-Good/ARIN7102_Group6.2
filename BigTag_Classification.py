from classification import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import joblib



class TfClassifier(Classifier):
    def __init__(self, type):
        super().__init__(type)
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  
            stop_words='english',  
            max_df=0.95,  
            min_df=2
            )
        self.classifer = None
    
    def load_data(self):
        df_customer = pd.read_csv('./dataset/Customer.csv')
        df_investment = pd.read_csv('./dataset/Funding_and_investment.csv')
        df_SCM = pd.read_csv('./dataset/Supply_Chain.csv')

        title_SCM = df_SCM['Title']
        content_SCM = df_SCM['Summary']

        title_investment = df_investment['Title']
        content_investment = df_investment['Summary']

        title_customer = df_customer['Title']
        content_customer = df_customer['Summary']
        words = []
        labels = []
        for content in title_SCM:
            words.append(content)
            labels.append(0)
        for content in content_SCM:
            words.append(content)
            labels.append(0)
        for content in title_investment:
            words.append(content)
            labels.append(1)
        for content in content_investment:
            words.append(content)
            labels.append(1)
        for content in title_customer:
            words.append(content)
            labels.append(2)
        for content in content_customer:
            words.append(content)
            labels.append(2)
        return words, labels

    def train(self):
        words, labels = self.load_data()
        self.vectorizer.fit(words)
        X = self.vectorizer.transform(words)
        classifier = OneVsRestClassifier(LinearSVC(random_state=7102))
        classifier.fit(X, labels)
        self.classifier = classifier
        joblib.dump(classifier, "model/tfmodel.pkl")
        joblib.dump(self.vectorizer, "model/tfvec.pkl")


    def predict(self, input_question):
        if not self.classifer:
            with open('./model/tfmodel.pkl', 'rb') as file:
                self.classifer = joblib.load(file)
            with open('./model/tfvec.pkl', 'rb') as file:
                self.vectorizer = joblib.load(file)
        
        embedding = self.vectorizer.transform(input_question)
        res = self.classifer.predict(embedding)[0]
        return res

