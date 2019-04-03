import re
import os
from collections import Counter

path_db = 'aclImdb/train/pos'
data_dic_pos = {}
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

            # Split using regular expression
            # 'a word character (a-z etc.) repeated one or more times'
            # A more complete solution will use a tokenizer (nltk for example)
            doc_tokenized = re.findall('\w+', doc)

            # Tokenize and count
            doc_bow = Counter(doc_tokenized)

            # Save in dictionary
            data_dic_pos[file] = doc_bow

        except:
            print('Error in reading doc')

        print(doc_bow)
