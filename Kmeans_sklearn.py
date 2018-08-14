from library import helpers
import time
from config import env
from library import prepare
from library import kmeans_info
from library import models
import joblib
import time
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
from config import env
from library import helpers
import os
import simplejson as json
import operator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from collections import Counter
from config import env

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# prepairing
articles = helpers.read_csv_column(env.APP_DIR + 'dataset.csv', 1)
timer = time.time()
articles = prepare.clear_list(articles)
print('[prepairing] Cleaned in %s' % (time.time() - timer))
timer = time.time()
articles = prepare.exclude_len(articles)
timer = time.time()
articles = prepare.stem_list(articles)
print('[prepairing] Stematized in %s' % (time.time() - timer))
joblib.dump(articles, env.APP_DIR + 'data/prepaired_yen.gz')
articles = joblib.load(env.APP_DIR + 'data/prepaired_yen.gz')
