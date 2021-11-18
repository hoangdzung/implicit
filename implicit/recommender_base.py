""" Base class for recommendation algorithms in this package """
import itertools
import multiprocessing
from abc import ABCMeta, abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class ModelFitError(Exception):
    pass


class RecommenderBase(object):
    """Defines the interface that all recommendations models here expose"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, user_items):
        """
        Trains the model on a sparse matrix of item/user/weight

        Parameters
        ----------
        user_items : csr_matrix
            A matrix of shape (number_of_users, number_of_items). The nonzero
            entries in this matrix are the items that are liked by each user.
            The values are how confident you are that the item is liked by the user.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
    ):
        """
        Recommends items for a user

        Calculates the N best recommendations for a user, and returns a list of itemids, score.

        Parameters
        ----------
        userid : int
            The userid to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (number_users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            calculate the best items for this user.
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specificed user.
        filter_items : sequence of ints, optional
            List of extra item ids to filter out from the output
        recalculate_user : bool, optional
            When true, don't rely on stored user state and instead recalculate from the
            passed in user_items

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays
        """
        pass

    @abstractmethod
    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        """
        Rank given items for a user and returns sorted item list.

        Parameters
        ----------
        userid : int
            The userid to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (number_users, number_items). This lets us
            (optionally) recalculate user factors (see `recalculate_user` parameter) as
            required
        selected_items : List of itemids
        recalculate_user : bool, optional
            When true, don't rely on stored user state and instead recalculate from the
            passed in user_items

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays. it only contains items that appears in
            input parameter selected_items
        """
        pass

    @abstractmethod
    def similar_users(self, userid, N=10):
        """
        Calculates a list of similar users

        Parameters
        ----------
        userid : int
            The row id of the user to retrieve similar users for
        N : int, optional
            The number of similar users to return

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays
        """
        pass

    @abstractmethod
    def similar_items(self, itemid, N=10, react_users=None, recalculate_item=False):

        """
        Calculates a list of similar items

        Parameters
        ----------
        itemid : int
            The row id of the item to retrieve similar items for
        N : int, optional
            The number of similar items to return
        react_users : csr_matrix, optional
            A sparse matrix of shape (number_items, number_users). This lets us look
            up the reacted users and their weights for the item.
        recalculate_item : bool, optional
            When true, don't rely on stored item state and instead recalculate from the
            passed in react_users

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays
        """
        pass
