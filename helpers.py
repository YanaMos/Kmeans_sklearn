import csv
import re
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

lancaster_stemmer = LancasterStemmer()


def read_csv_column(file, column_id):
    result = []
    data = read_csv(file)
    for row in data:
        if len(row) - 1 >= column_id:
            result.append(row[column_id])
    return result


def read_csv(file):
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)


def clear_list(data):
    for i, s in enumerate(data):
        s = s.lower()
        s = re.sub(r'[^a-zA-Z_ ]', '', s)
        data[i] = re.sub(r'\s+', ' ', s)
    return data


def stop_words_exclude(data):
    result = []
    for row in data:
        text = [word for word in row.split(' ') if word not in stopwords.words('english')]
        result.append(' '.join(text))
    return result

def exclude_len(data):
    data = [s for i, s in enumerate(data) if len(s) > 500]
    return data


def stem_word(word):
    stem_word_cache = {}
    s = word[0]
    if not s in stem_word_cache:
        stem_word_cache[s] = {}

    if word in stem_word_cache[s]:
        return stem_word_cache[s][word]
    stem_word_cache[s][word] = lancaster_stemmer.stem(word)
    return stem_word_cache[s][word]


def stem_list(data):
    result = []
    for i, sentence in enumerate(data):
        words = []
        for word in sentence.split(' '):
            if not word:
                continue
            words.append(stem_word(word))
        result.append(' '.join(words))
    return result
