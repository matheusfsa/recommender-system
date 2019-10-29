from sklearn.feature_extraction.text import TfidfTransformer

def tf_id(X, movies=None):
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    X= tfidf_transformer.transform(X).toarray()
    return X