import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

path_db = 'aclImdb/train/pos'
doc_list = []
for subdir, dirs, files in os.walk(path_db):
    for file in files:
        # Print file name
        print(file)

        # Read csv file
        path_full = os.path.join(subdir, file)
        try:
            # Read lines into list
            lines_list = open(path_full, "r").readlines()

            # Join lines together into one element (doc)
            doc = ' '.join(lines_list).lower()

            # Append to list (corpus)
            doc_list.append(doc)

        except:
            print('Error in reading doc')

vectorizer = TfidfVectorizer()
tf_idf_scores = vectorizer.fit_transform(doc_list)
example_index = 99
example_doc = doc_list[example_index]
print(example_doc)
ind_important0 = tf_idf_scores.toarray()[example_index, :].argsort()[-1]
ind_important1 = tf_idf_scores.toarray()[example_index, :].argsort()[-2]
ind_important2 = tf_idf_scores.toarray()[example_index, :].argsort()[-3]
doc_tokenized = re.findall('\w+', example_doc)
words_in_corpus = vectorizer.get_feature_names()
print('first most important word: ' + words_in_corpus[ind_important0])
print('second most important word: ' + words_in_corpus[ind_important1])
print('third most important word: ' + words_in_corpus[ind_important2])


