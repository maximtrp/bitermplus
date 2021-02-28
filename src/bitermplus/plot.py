__all__ = ['vis_prepare_model']
from pyLDAvis import prepare as plv_prepare
import numpy as np


def vis_prepare_model(
        ttd: np.ndarray,
        dtd: np.ndarray,
        docs_len: np.ndarray,
        vocab: np.ndarray,
        tf: np.ndarray,
        **kwargs: dict
        ):
    """Simple wrapper around `pyLDAvis.prepare` method.

    Parameters
    ----------
    ttd : np.ndarray
        Matrix of topic-term probabilities. Where `n_terms` is `len(vocab)`.
        (n_topics, n_terms)
    dtd : np.ndarray
        Matrix of document-topic probabilities.  shape (n_docs, n_topics)
    docs_len : np.ndarray
        The length of each document, i.e. the number of words in each document.
        The order of the numbers should be consistent with the ordering of the
        docs in `doc_topic_dists`.  shape n_docs
    vocab : np.ndarray
        List of all the words in the corpus used to train the model.  shape
        n_terms
    tf : np.ndarray
        The count of each particular term over the entire corpus. The ordering
        of these counts should correspond with `vocab` and `topic_term_dists`.
        shape n_terms
    **kwargs : dict
        Keyword arguments passed to `pyLDAvis.prepare` method.

    Returns
    -------
    data : PreparedData
        Output of `pyLDAvis.prepare` method.
    """

    vis_data = plv_prepare(
        ttd, dtd, docs_len, vocab, tf, **kwargs)
    return vis_data
