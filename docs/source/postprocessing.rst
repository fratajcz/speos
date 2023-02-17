speos.postproscessing
=====================

The Preprocessor handles everything that happens after the crossvalidation ensemble has been trained. It gathers the candidate genes and conducts several external validation tasks among the candidates.
Usually, the same external validation tasks are also performed on the positively labeled genes so that the user can judge how well his or her candidate genes compare to the 'gold standard' positives.

If the crossvalidation pipeline has been chosen, these tasks are run automatically and the user does not need to bother with this class.
If, however, the user needs more detailed results, there is an example at the end of this page which shows how to obtain them (TODO).

For a more detailed of description of the the external validation tasks you can consult the method section of the `accompanying paper <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1>`_ .

.. autoclass:: speos.postprocessing.postprocessor.PostProcessor
    :members:
    :inherited-members: