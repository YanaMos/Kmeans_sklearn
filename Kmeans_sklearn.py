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

    def buil_model(self, text):

        # model
        true_k = 8
        matrix, vectorizer, vector, vocabulary = models.build_vectors_yen(text)
        model = models.build_kmeans(matrix, vocabulary, true_k)

        kmeans_info.cluster_value(model)
        kmeans_info.top_terms_cluster(model, vectorizer, true_k)

    def build_vectors(articles):
        time_start = time.time()
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=100, use_idf=True)
        vec = vectorizer.fit(articles)
        matrix = vectorizer.transform(articles)

        # Fixing vocabulary_ from numpy.int64
        vocabulary = {}
        for word, count in vectorizer.vocabulary_.items():
            vocabulary[int(count)] = word

        print('vectors in %s' % (time.time() - time_start))
        return [matrix, vectorizer, vec, vocabulary]

    def build_kmeans(matrix, vocabulary, true_k):
        time_start = time.time()

        # Making list from vocabulary for hashing
        vocabulary_list = []
        for id, word in vocabulary.items():
            vocabulary_list.append(word + '|' + str(id))
        vocabulary_list.sort()
        vocabulary_hash = helpers.hash_crc32(json.dumps(vocabulary_list))
        print('model hash: ' + vocabulary_hash)

        # Building unique cache file name
        cache_file = env.APP_DIR + 'cache/model_' + str(true_k) + '_' + vocabulary_hash + '.gz'

        # Check if already cached
        if os.path.exists(cache_file):
            return joblib.load(cache_file)

        # Build model
        model = KMeans(n_clusters=true_k, init='k-means++', n_init=10, max_iter=300, n_jobs=-1)
        model.fit(matrix)

        # Caching
        joblib.dump(model, cache_file, 9)
        print('model in %s' % (time.time() - time_start))
        return model

  