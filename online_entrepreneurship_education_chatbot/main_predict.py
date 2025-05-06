from BigTag_Classification import TfClassifier
from SmallTag_Classification import KNNClassification
from EmbeddingClassification import EmbeddingClassification
import pandas as pd
class All_predict:
    def __init__(self):
        dic1 = {}
        dic2 = {}
        dic1[2] = pd.read_csv('./dataset/Customer.csv')
        dic1[1] = pd.read_csv('./dataset/Funding_and_Investment.csv')
        dic1[0] = pd.read_csv('./dataset/Supply_Chain.csv')
        dic2[2] = pd.read_csv('./dataset/cluster_output_customer.csv')
        dic2[1] = pd.read_csv('./dataset/cluster_output_investment.csv')
        dic2[0] = pd.read_csv('./dataset/cluster_output_SCM.csv') 
        self.dic1 = dic1
        self.dic2 = dic2
    def getResult(self, input_question):
        input_questions = [input_question]
        tfClassifier = TfClassifier("tf")
        result1 = tfClassifier.predict(input_questions)
        knnClassifer = KNNClassification(k=3)
        result2 = knnClassifer.predict(input_questions, result1)

        content = self.dic1[result1]
        content2 = self.dic2[result1]
        res = ""
        res += "your question seems relevant to the topic : "
        if result1 == 0:
            ttt = "Supply Chain Management"
        if result1 == 1:
            ttt = "Funding and Investment Strategy"
        if result1 == 2:
            ttt = "Customer Analytics"
        res += ttt
        res += "<br><br>"
        titles = content2['Titles'].fillna('').values.tolist()
        contents = content['Summary'].fillna('').values.tolist()
        titles2 = content['Title'].fillna('').values.tolist()
        url = content['URL'].fillna('').values.tolist()
        dic = {}
        dic_URL = {}
        for i in range(len(titles2)):
            dic[titles2[i]] = contents[i]
            dic_URL[titles2[i]] = url[i]
        sentences = []
        for i in range(len(titles)):
            if i != result2:
                continue
            title_list = titles[i].split(",")
            print("****************")
            for tmp in title_list:
                sentences.append(tmp.strip() + "+" + dic[tmp.strip()])
            embeddingClassification = EmbeddingClassification("embedding")
            similar_sentences = embeddingClassification.find_most_similar(input_questions, sentences)
            cc = 0
            for sentence, score in similar_sentences:
                if cc > 2:
                    break
                # print(sentence)
                if cc == 0:
                    res += sentence.split("+")[0]
                    res += "<br>"
                    res += dic[sentence.split("+")[0]]
                    res += "<br>"
                    res += dic_URL[sentence.split("+")[0]]
                    res += "<br><br>"
                    res += "Here is the some answer you may interested in:<br>"
                else:
                    res += sentence.split("+")[0]
                    res += ": "
                    res += dic_URL[sentence.split("+")[0]]
                    res += "<br>"


                # print(dic_URL[sentence.split("+")[0]])
                cc += 1
            return res

