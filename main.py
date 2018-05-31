from util import *
from models import *

def main():
	x_tr, y_tr, x_val, y_val, x_te, y_te = get_dataset("data/SMSSpamCollection")
	vec = create_feature_extractor(x_tr, ngram=1)

	# Perform transformations on input data
	x_tr_ngram = vec.transform(x_tr)
	x_val_ngram = vec.transform(x_val)
	x_te_ngram = vec.transform(x_te)

	multinomialnb_main(x_tr_ngram, y_tr, x_val_ngram, \
	                   y_val, x_te_ngram, y_te)

if __name__ == "__main__":
    main()