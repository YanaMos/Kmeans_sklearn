import re
from collections import Counter
import matplotlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import helpers

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class KmeansSklearn:
    def __init__(self, data):
        self.data = self.text_cleaning(data)
        self.handle()

    def handle(self):
        true_k = 4
        model, matrix, vectorizer, true_k = self.buil_model(self.data, true_k)
        self.cluster_value(model)
        self.top_terms_cluster(model, vectorizer, true_k)
        self.sil_coef(matrix, model)
        self.cluster(self.data, model, true_k)

    def text_cleaning(self, data):
        data = helpers.clear_list(data)
        data = helpers.stop_words_exclude(data)
        data = helpers.exclude_len(data)
        data = helpers.stem_list(data)
        return data

    def buil_model(self, text, true_k):
        matrix, vectorizer, vector, vocabulary = self.build_vectors(text)
        model = self.build_kmeans(matrix, true_k)
        return model, matrix, vectorizer, true_k

    def build_vectors(self, articles):
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3, use_idf=True)
        vec = vectorizer.fit(articles)
        matrix = vectorizer.transform(articles)

        # Fixing vocabulary_ from numpy.int64
        vocabulary = {}
        for word, count in vectorizer.vocabulary_.items():
            vocabulary[int(count)] = word

        return [matrix, vectorizer, vec, vocabulary]

    def build_kmeans(self, matrix, true_k):
        model = KMeans(n_clusters=true_k, init='k-means++', n_init=10, max_iter=300, n_jobs=-1)
        model.fit(matrix)
        return model

    def cluster_value(self, model):
        clusters = model.labels_
        print("value {0}".format(Counter(clusters).values()))
        print("keys {0}".format(Counter(clusters).keys()))

    def top_terms_cluster(self, model, vectorizer, true_k):
        print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print()

    def sil_coef(self, matrix, model):
        for n_cluster in range(2, 28):
            label = model.labels_
            sil_coeff = silhouette_score(matrix, label, metric='euclidean')
            print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

    def cluster(self, articles, model, true_k):
        pipeline = Pipeline([('vect', TfidfVectorizer(analyzer='word', min_df=5, use_idf=True)), ])
        Z = pipeline.fit_transform(articles).todense()
        clusters = model.labels_
        print("count titles per cluster {0}".format(Counter(clusters).values()))

        LABEL_COLOR_MAP = {
            0: '#a8bb19',
            1: '#7cb9e8',
            2: '#b284be',
            3: '#00308f'
        }

        label_color = [LABEL_COLOR_MAP[l] for l in clusters]

        pca = PCA(n_components=true_k).fit(Z)
        data2D = pca.transform(Z)
        plt.scatter(data2D[:, 0], data2D[:, 1], c=label_color)
        plt.savefig('/Users/yanamosiichuk/www/git_private/Kmeans_sklearn/cluster_visual.png')
