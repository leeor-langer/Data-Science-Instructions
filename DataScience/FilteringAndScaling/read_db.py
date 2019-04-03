import os

path_db = 'aclImdb/train/pos'
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

            print(doc)
        except:
            print('Error in reading doc')


