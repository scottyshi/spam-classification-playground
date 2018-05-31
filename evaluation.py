import sklearn.metrics as met

"""
	TODO: additional evaluation metrics to be used?
"""
def print_metrics(y_pred, y_true):
	print(met.accuracy_score(y_true, y_pred))
	print(met.classification_report(y_true, y_pred))
	print(met.precision_recall_fscore_support(y_true, y_pred))
