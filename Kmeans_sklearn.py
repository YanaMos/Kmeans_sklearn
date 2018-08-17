from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
import simplejson as json
import operator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Kmeans_sklearn:
    def handle(self):
        text = self.read_csv_column('dataset.csv', 1)
        self.text_cleaning(text)

    def text_cleaning(self, text):
        text = self.clear_list(text)
        text = self.exclude_len(text)
        text = self.stem_list(text)
        return text

    def read_csv_column(self, file, column_id):
        result = []
        data = read_csv(file)
        for row in data:
            if len(row) - 1 >= column_id:
                result.append(row[column_id])
        return result

    def clear_list(self, data):
        for i, s in enumerate(data):
            s = s.lower()
            s = re.sub(r'[^a-zA-Z_ ]', '', s)
            data[i] = re.sub(r'\s+', ' ', s)
        return data

    def exclude_len(self, data):
        data = [s for i, s in enumerate(data) if len(s) > 500]
        return data

    def stem_list(self, data):
        result = []
        for i, sentence in enumerate(data):
            words = []
            for word in sentence.split(' '):
                if not word:
                    continue
                words.append(stem_word(word))
            result.append(' '.join(words))
        return result


