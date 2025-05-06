from BigTag_Classification import TfClassifier
from SmallTag_Classification import KNNClassification
from EmbeddingClassification import EmbeddingClassification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class ModelEvaluator:
    def __init__(self):
        # 加载数据集
        self.datasets = {
            0: {'data': pd.read_csv('./data_test/Supply_Chain_test.csv'), 'name': 'Supply Chain Management'},
            1: {'data': pd.read_csv('./data_test/Funding_and_Investment_test.csv'), 'name': 'Funding and Investment Strategy'},
            2: {'data': pd.read_csv('./data_test/Customer_test.csv'), 'name': 'Customer Analytics'}
        }
        
        # 加载聚类结果
        self.cluster_outputs = {
            0: pd.read_csv('./dataset/cluster_output_SCM.csv'),
            1: pd.read_csv('./dataset/cluster_output_investment.csv'),
            2: pd.read_csv('./dataset/cluster_output_customer.csv')
        }
        
        # 初始化模型
        self.tf_classifier = TfClassifier("tf")
        
    def evaluate_tf_classification(self):
        """评估TF分类器的准确性"""
        true_labels = []
        predicted_labels=[]
        
        # 准备测试数据
        for label, data in self.datasets.items():
            # 使用Title作为测试问题，每类取20个样本或全部样本（如果少于20）
            samples = data['data']['question'].dropna().sample(min(20, len(data['data'])), random_state=42)
            for question in samples:
                #questions.append(question)
                #print(label)
                predicted_label=self.tf_classifier.predict([question])
                predicted_labels.append(predicted_label)
                #print(predicted_label)
                true_labels.append(label)
        
        # 进行预测
        #predicted_labels = self.tf_classifier.predict([questions])
        print(predicted_labels)
        
        # 确保预测结果是数组形式
        if not isinstance(predicted_labels, (list, np.ndarray)):
            predicted_labels = [predicted_labels] if isinstance(predicted_labels, (int, float)) else []
        
        # 检查长度是否匹配
        if len(true_labels) != len(predicted_labels):
            print(f"Warning: Length mismatch - true_labels: {len(true_labels)}, predicted_labels: {len(predicted_labels)}")
            min_len = min(len(true_labels), len(predicted_labels))
            true_labels = true_labels[:min_len]
            predicted_labels = predicted_labels[:min_len]
        
        # 如果没有任何预测结果，返回0
        if len(predicted_labels) == 0:
            print("Error: No predictions returned by tf_classifier.predict()")
            return 0, "No predictions available"
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=[d['name'] for d in self.datasets.values()])
        
        print(f"TF Classifier Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        return accuracy, report
    
    
    def full_evaluation(self):
        """执行完整评估流程"""
        print("Starting Model Evaluation...\n")
        
        results = {}
        
        try:
            # 评估TF分类器
            tf_accuracy, tf_report = self.evaluate_tf_classification()
            results['tf_accuracy'] = tf_accuracy
            results['tf_report'] = tf_report
        except Exception as e:
            print(f"Error evaluating TF classifier: {str(e)}")
            results['tf_error'] = str(e)
        
        

def main():
    evaluator = ModelEvaluator()
    results = evaluator.full_evaluation()
    return results

if __name__ == "__main__":
    main()