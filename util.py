from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

"""
    @in - path: path to raw dataset. label and sentence are separated
        by a tab
    @out: LIST containing [xtrain, ytrain, xval, yval, xtest,
        ytest]
    
    Reads the raw dataset and splits into train/cross-val/test
"""
def get_dataset(path):
    with open(path) as f:
        raw_ds = f.readlines()

    x_raw = []
    y_raw = []

    for l in raw_ds:
        label, sentence = l.split("\t", 1)
        y_raw.append(0 if label == "ham" else 1)
        x_raw.append(sentence.rstrip())

    x_t, x_te, y_t, y_te = train_test_split(x_raw, \
                                            y_raw, \
                                            test_size=0.2, \
                                            random_state=1)
    x_tr, x_val, y_tr, y_val = train_test_split(x_t, \
                                               y_t, \
                                               test_size=0.2, \
                                               random_state=1)
    return x_tr, y_tr, x_val, y_val, x_te, y_te


"""
    @in - x_train: train dataset i want to learn vocabulary of
    @in - ngram: value of gram i want to use for feature extraction
    @out - OBJECT containing vectorizer built from training dataset
        available to transform other inputs. 
        
    This function will learn the corpus's vocabulary using a specified
        value ngram. TODO: try different features to use
"""
def create_feature_extractor(x_train, ngram = 2):
    vec = CountVectorizer(ngram_range=(ngram,ngram))
    vec.fit(x_train)
    return vec
    