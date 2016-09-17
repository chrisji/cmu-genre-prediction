from nltk import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from pylab import barh, plot, yticks, show, grid, xlabel, ylabel, figure
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import NMF, LatentDirichletAllocation


def nmf_topics(summary, n_topics=2, n_top_words=10, n_gram=(1, 3)):
    """
    Use NMF to extract the topics from a summary (uses tf-idf)
    :param summary:
    :param n_topics:
    :param n_top_words:
    :param n_gram:
    :return:
    """
    # Fit the vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',
                                       ngram_range=n_gram)
    tfidf = tfidf_vectorizer.fit_transform(summary.split('.'))

    # Fit the NMF model
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)


def lda_topics(summary, n_topics=2, n_top_words=10, n_gram=(1, 3)):
    """
    Use LDA to extract the topics from a summary (uses raw counts)
    :param summary:
    :param n_topics:
    :param n_top_words:
    :param n_gram:
    :return:
    """
    # Fit the vectorizer
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english',
                                    ngram_range=n_gram)
    tf = tf_vectorizer.fit_transform(summary.split('.'))

    # Fit the LDA model
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)


def visualize_chi2(summaries, genres, n_gram=(1, 3)):
    """
    Visualize the most discriminative features for each genre
    :param summaries:
    :param genres:
    :param n_gram:
    :return:
    """
    vectorizer = TfidfVectorizer(ngram_range=n_gram, lowercase=True,
                                 norm=None, smooth_idf=True, sublinear_tf=True)
    new_summaries = []
    new_genres = []
    for (summary, genre) in (summaries, genres):
        for sentence in summary.split('.'):
            new_summaries.append(sentence)
            new_genres.append(genre)
    X_train = vectorizer.fit_transform(new_summaries)
    chi2score = chi2(X_train, new_genres)[0]
    figure(figsize=(6, 6))
    wscores = zip(vectorizer.get_feature_names(), chi2score)
    wchi2 = sorted(wscores, key=lambda x: x[1])
    topchi2 = zip(*wchi2[-20:])
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    barh(x, topchi2[1], align='center', alpha=.2, color='g')
    plot(topchi2[1], x, '-o', markersize=2, alpha=.8, color='g')
    yticks(x, labels)
    xlabel('$\chi^2$')
    ylabel('Top discriminative features')
    show()


def print_top_words(model, feature_names, n_top_words):
    """
    Print the top words for a document, using a specific model
    :param model:
    :param feature_names:
    :param n_top_words:
    :return:
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
