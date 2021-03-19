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
    """Simple wrapper around :meth:`pyLDAvis.prepare` method.

    Parameters
    ----------
    ttd : np.ndarray
        Topics vs words probabilities matrix (T x W).
    dtd : np.ndarray
        Document vs topics probabilities (D x T).
    docs_len : np.ndarray
        The length of each document, i.e. the number of words in each document.
        The order of the numbers should be consistent with the ordering of the
        docs in `dtd` (D x 1).
    vocab : np.ndarray
        List of all the words in the corpus used to train the model (W x 1).
    tf : np.ndarray
        The count of each particular term over the entire corpus (W x 1).
    **kwargs : dict
        Keyword arguments passed to :meth:`pyLDAvis.prepare` method.

    Returns
    -------
    data : PreparedData
        Output of :meth:`pyLDAvis.prepare` method.
    """

    vis_data = plv_prepare(
        ttd, dtd, docs_len, vocab, tf, **kwargs)
    return vis_data
