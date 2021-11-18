""" Base class for recommendation algorithms in this package """
import multiprocessing
import warnings
from abc import ABCMeta, abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm.auto import tqdm

from ..recommender_base import RecommenderBase
from .topk import topk

""" 
TODO's
    update RecommenderBase signature
    docstring for parameters (items/userids etc)
    gpu: filter options
    cpu topk: get appropiate batch size ?
    cpu topk: rename to be consistent with gpu?
    cpu recommend: remap filter items when items passed in? (OR: not allow both at the same time)
    cpu recommend: batch mode recalculate users
    gpu bpr: does this work? seems like we're getting a item_users matrix
    filter options for similar_items/similar_users
"""


class MatrixFactorizationBase(RecommenderBase):
    """MatrixFactorizationBase contains common functionality for recommendation models.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(self):
        # learned parameters
        self.item_factors = None
        self.user_factors = None

        # cache of user, item norms (useful for calculating similar items)
        self._user_norms, self._item_norms = None, None

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        user = self._user_factor(userid, user_items, recalculate_user)

        filter_query_items = None
        if filter_already_liked_items:
            filter_query_items = user_items[userid]

        item_factors = self.item_factors
        if items is not None:
            items = np.array(items)
            # check selected items are in the model
            if items.max() >= self.item_factors.shape[0] or items.min() < 0:
                raise IndexError("Some itemids are not in the model")
            item_factors = item_factors[items]

            # TODO: we will need to remap filter_query_items, filter_items

        ids, scores = topk(
            self.item_factors if items is None else self.item_factors[items],
            user,
            N,
            filter_query_items=filter_query_items,
            filter_items=filter_items,
        )

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        # if given a
        if items is not None:
            ids = items[ids]

        return ids, scores

    def recommend_all(
        self,
        user_items,
        N=10,
        recalculate_user=False,
        filter_already_liked_items=True,
        filter_items=None,
        num_threads=0,
        show_progress=True,
        batch_size=0,
        users_items_offset=0,
    ):
        warnings.warn(
            "recommend_all is deprecated. Use recommend with an array of userids instead",
            DeprecationWarning,
        )

        userids = np.arange(user_items.shape[0]) + users_items_offset
        if users_items_offset:
            adjusted = lil_matrix(
                (user_items.shape[0] + users_items_offset, user_items.shape[1]),
                dtype=user_items.dtype,
            )
            adjusted[users_items_offset:] = user_items
            user_items = adjusted.tocsr()

        ids, scores = self.recommend(
            userids,
            user_items,
            N=N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )
        return ids

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        warnings.warn(
            "rank_items is deprecated. Use recommend with the 'items' parameter instead",
            DeprecationWarning,
        )
        return self.recommend(
            userid,
            user_items,
            recalculate_user=recalculate_user,
            items=selected_items,
            filter_already_liked_items=False,
        )

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if recalculate_user:
            return self.recalculate_user(userid, user_items)
        else:
            return self.user_factors[userid]

    def _item_factor(self, itemid, react_users, recalculate_item=False):
        if recalculate_item:
            return self.recalculate_item(itemid, react_users)
        else:
            return self.item_factors[itemid]

    def recalculate_user(self, userid, user_items):
        raise NotImplementedError("recalculate_user is not supported with this model")

    def recalculate_item(self, itemid, react_users):
        raise NotImplementedError("recalculate_item is not supported with this model")

    def similar_users(self, userid, N=10):
        factor = self.user_factors[userid]
        factors = self.user_factors
        norms = self.user_norms
        norm = norms[userid]
        return self._get_similarity_score(factor, norm, factors, norms, N)

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(self, itemid, N=10, react_users=None, recalculate_item=False):
        factor = self._item_factor(itemid, react_users, recalculate_item)
        factors = self.item_factors
        norms = self.item_norms
        if recalculate_item:
            # TODO: batch mode recalculate
            norm = np.linalg.norm(factor)
            norm = norm if norm != 0 else 1e-10
        else:
            norm = norms[itemid]

        return self._get_similarity_score(factor, norm, factors, norms, N)

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

    def _get_similarity_score(self, factor, norm, factors, norms, N):
        ids, scores = topk(factors, factor, N, item_norms=norms)
        if np.isscalar(norm):
            ids, scores = ids[0], scores[0]
            scores /= norm
        else:
            scores /= norm[:, None]
        return ids, scores

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = np.linalg.norm(self.user_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._user_norms[self._user_norms == 0] = 1e-10
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

    def _check_fit_errors(self):
        is_nan = np.any(np.isnan(self.user_factors), axis=None)
        is_nan |= np.any(np.isnan(self.item_factors), axis=None)
        if is_nan:
            raise ModelFitError("NaN encountered in factors")
