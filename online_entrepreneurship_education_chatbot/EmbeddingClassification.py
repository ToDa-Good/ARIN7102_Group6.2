from classification import Classifier
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

class EmbeddingClassification(Classifier):
    def __init__(self, type):
        #self.embedding_model = hub.load("/Users/Kenda/Downloads/elmo-tensorflow1-elmo-v2")
        self.embedding_model = hub.load("./elmo-tensorflow1-elmo-v2")
    def get_embedding(self, sentences):   
        embeddings = self.embedding_model.signatures['default'](tf.constant(sentences))['default']
        return embeddings.numpy()
    
    def calculate_similarity(self, sentence1, sentence2):
        
        emb1 = self.get_embedding(sentence1)
        emb2 = self.get_embedding(sentence2)

        similarity = 1 - cosine(emb1[0], emb2[0])
        
        return similarity

    def find_most_similar(self, target_sentence, sentence_list):
        target_emb = self.get_embedding(target_sentence)
        similarities = []
        
        all_embeddings = self.get_embedding(sentence_list)
        
        for i, emb in enumerate(all_embeddings):
            similarity = 1 - cosine(target_emb[0], emb)
            similarities.append((sentence_list[i], similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
def main():
    test_texts = [
        "What is the importance of understanding evolving consumer behavior for brands?",
        "How can you find trending products using Google Trends?",
        "What methods can be used to measure brand awareness?"
    ]
    embeddingClassification = EmbeddingClassification("embedding")
    print(embeddingClassification.find_most_similar(["What is the importance of understanding evolving consumer behavior for brands?"], test_texts))

if __name__ == "__main__":
    main()
