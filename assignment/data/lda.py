# assignment/data/lda.py

import numpy as np
from gensim.models import LdaModel

def get_document_topic_matrix(lda_model: LdaModel, bow_corpus: list) -> np.ndarray:
    """
    Get the document-topic matrix from an LDA model.

    Args:
        lda_model (LdaModel): Trained LDA model.
        bow_corpus (list): Bag-of-words corpus.

    Returns:
        np.ndarray: Document-topic matrix.
    """

    topic_distributions = lda_model.get_document_topics(bow_corpus)

    # The topic distributions are sparse coordinates, so we'll convert them to a matrix
    sparse_dist = np.zeros((len(topic_distributions), len(lda_model.get_topics())))
    for i, doc in enumerate(topic_distributions):
        for topic, score in doc:
            sparse_dist[i][topic] = score

    return sparse_dist
