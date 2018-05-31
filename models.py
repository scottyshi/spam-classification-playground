import time
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as met
import numpy as np

from evaluation import *

""" MULTINOMIAL NAIVE BAYES """

"""
    @in - trainset and x input (either validation or test)
    @out - predicted validation values and hyperparameter used
    
    Trains and returns validation loss
"""
def multinomialnb_run(x_tr_ngram, y_tr, \
                      x_in_ngram, a, mode="train"):
    model = MultinomialNB(alpha = a)
    model.fit(x_tr_ngram, y_tr)
    start = time.time() # Used for measuring decoding speed
    y_p = model.predict(x_in_ngram)
    end = time.time()
    if mode == "test":
        print("Decoding time: ", end-start)
    return y_p, a


"""
    Top level code to train a multinomial naive bayes model
"""
def multinomialnb_main(x_tr_ngram, y_tr, \
                       x_val_ngram, y_val, \
                       x_te_ngram, y_te):
    # TODO: for now, i'll just use gridsearch, later i'll implement
    #   random search.
    hyperparam_searchspace = np.logspace(-1,0,7)
    top_val_acc = 0
    best_hyp = None
    for a in hyperparam_searchspace:
        y_p, alpha = multinomialnb_run(x_tr_ngram, y_tr, \
                                       x_val_ngram, a)
        val_acc = met.accuracy_score(y_val, y_p)
        if val_acc > top_val_acc:
            top_val_acc = val_acc
            best_hyp = alpha
            
    print("Crossval finished -- highest val accuracy: ",\
             top_val_acc)
    print("Best hyperparameter selected: ", best_hyp)
    
    # Testing pass
    y_p, _ = multinomialnb_run(x_tr_ngram, y_tr, \
                               x_te_ngram, a, mode="test")
    print_metrics(y_p, y_te)
            