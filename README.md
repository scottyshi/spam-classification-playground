# Intro
Here I will play with several different models for the task of spam classification of text messages. Here are the models I've tried so far:
* Multinomial Naive Bayes

# Dataset Description
* UCI SMS Spam Collection Dataset (public dataset)
* 5574 examples // 4827 real (86.6%) // 747 spam (13.4%)
   * 3375 examples are from NUS SMS Corpus (Singaporean english) -- need to account for language difference??
* I used a 0.64/0.16/0.2 train/val/test split

# Model Descriptions
* Multinomial Naive Bayes
   * This is the simplest model I wanted to try first. I wanted to use MultinomialNB over BernoulliNB since BernoulliNB restricts the features to be distributed bernoulli (this may not be desirable since some words can be present in either spam/not spam)
   * The hyperparameter I tuned for MultinomialNB is "alpha" which helps calculate a smoothed version of MLE -- I used gridsearch to select this

# Results Table
| Model | Acc | Spam-prec | Spam-Recall | Real-Prec | Real-Recall | Decoding Time |
| ----- | ----- | ------ | ----- | ------ | ----- | ----- |
| Multinomial NB (bi-gram) | 0.9839 | 0.96 | 0.91 | 0.99 | 0.99 | $$3.65*10^{-4}$$ |
| Multinomial NB (unigram) | 0.9901 | 0.96 | 0.96 | 0.99 | 0.99 | $$2.02*10^{-4}$$ |