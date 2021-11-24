""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/

This code will automically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html.
"""
import argparse
import codecs
import logging
import time

import numpy as np
import tqdm
import os 

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

# maps command line model argument to class name
MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "faiss_als": FaissAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
}

def get_techland(path='./techland.json'):
    assert os.path.isfile(path), "File not exist"
    from scipy.sparse import csr_matrix
    import json 

    data = json.load(open(path))
    company_to_id = {company: idx for idx, company in enumerate(data)}
    term_to_id = {}
    com_idxs = []
    term_idxs = []
    rates = []
    for com, term_to_score in tqdm.tqdm(data.items()):
        com_idx = company_to_id[com]
        for term, score in term_to_score.items():
            com_idxs.append(com_idx)
            try:
                term_idx = term_to_id[term]
            except KeyError as e:
                term_idx = len(term_to_id)
                term_to_id[term] = term_idx
            term_idxs.append(term_idx)
            rates.append(score)

    id_to_company = {idx: company for company, idx in company_to_id.items()}
    id_to_term = {idx: term for term, idx in term_to_id.items()}
    companies = np.array([str.encode(id_to_company[i]) for i in range(len(id_to_company))])
    terms = np.array([str.encode(id_to_term[i]) for i in range(len(id_to_term))])
    ratings = csr_matrix((rates, (com_idxs, term_idxs)), shape=(len(companies), len(terms)))

    return companies, terms, ratings

def get_model(model_name):
    print("getting model %s" % model_name)
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if model_name.endswith("als"):
        params = {"factors": 64, "dtype": np.float32}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def calculate_similar_artists(output_filename, model_name="als", data='lastfm'):
    """generates a list of similar artists in lastfm by utilizing the 'similar_items'
    api of the models"""
    if data == 'lastfm':
        artists, users, plays = get_lastfm()
    elif data == 'techland':
        artists, users, plays = get_techland()
    else:
        raise NotImplementedError
    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if model_name.endswith("als"):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # write out similar artists by popularity
    start = time.time()
    logging.debug("calculating top artists")

    user_count = np.ediff1d(plays.indptr)
    to_generate = sorted(np.arange(len(artists)), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    logging.debug("writing similar items")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for artistid in to_generate:
                artist = artists[artistid]
                for other, score in model.similar_items(artistid, 11):
                    o.write("%s\t%s\t%s\n" % (artist, artists[other], score))
                progress.update(1)

    logging.debug("generated similar artists in %0.2fs", time.time() - start)


def calculate_recommendations(output_filename, model_name="als", data='lastfm'):
    """Generates artist recommendations for each user in the dataset"""
    # train the model based off input params
    if data == 'lastfm':
        artists, users, plays = get_lastfm()
    elif data == 'techland':
        artists, users, plays = get_techland()
    else:
        raise NotImplementedError
    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if model_name.endswith("als"):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # generate recommendations for each user and write out to a file
    start = time.time()
    user_plays = plays.T.tocsr()
    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for userid, username in enumerate(users):
                for artistid, score in model.recommend(userid, user_plays):
                    o.write("%s\t%s\t%s\n" % (username, artists[artistid], score))
                progress.update(1)
    logging.debug("generated recommendations in %0.2fs", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates similar artists on the last.fm dataset"
        " or generates personalized recommendations for each user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similar-artists.tsv",
        dest="outputfile",
        help="output file name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="als",
        dest="model",
        help="model to calculate (%s)" % "/".join(MODELS.keys()),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="lastfm",
        dest="data",
        help="data to calculate lastfm or techland",
    )
    parser.add_argument(
        "--recommend",
        help="Recommend items for each user rather than calculate similar_items",
        action="store_true",
    )
    parser.add_argument(
        "--param", action="append", help="Parameters to pass to the model, formatted as 'KEY=VALUE"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculate_recommendations(args.outputfile, model_name=args.model, data = args.data)
    else:
        calculate_similar_artists(args.outputfile, model_name=args.model, data = args.data)
