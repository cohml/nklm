"""
Compute the type-token ratio (TTR) for each word type across all articles.

In other words, identify each unique word type, then compute the percentage of all
tokens across all articles that belong to each type. So given an text "The dog ate the
toy.", the TTR for the word "the" (case-insensitive) is 2 / 5 = 0.4.
"""

import argparse
import pandas as pd
import spacy

from pathlib import Path
from spacy.language import Language
from spacy.tokens import Token
from tqdm import tqdm
from typing import List

from nklm.util.defaults import DEFAULTS
from nklm.util.utils import full_path


def get_all_lemmas(articles: pd.Series, nlp: Language) -> List[Token]:
    """
    Extract list of all lemmatized tokens across all essays, excluding stop words,
    whitespace, and punctuation.

    Parameters
    ----------
    articles : pd.Series
        article bodies
    nlp : Language
        language model (i.e., ``en_core_web_lg``)

    Returns
    -------
    lemmas : List[Token]
        list of all lemmatized tokens across all essays, excluding stop words,
        whitespace, and punctuation
    """

    lemmas = []
    docs = nlp.pipe(articles, batch_size=16)

    for doc in tqdm(docs, total=articles.size):
        for token in doc:

            if any([token.is_stop, token.is_space, token.is_punct]):
                continue

            lemmas.append(token.lemma_)

    return lemmas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Get list of unique tokens used across all articles (and their '
                    'percentages out of all tokens), excluding stop words, '
                    'whitespace, and punctuation, writing the results to CSV.'
    )
    parser.add_argument(
        '-a', '--articles_csv_filepath',
        type=full_path,
        required=True,
        help='path to CSV containing article raw texts'
    )
    parser.add_argument(
        '-o', '--output_directory',
        type=full_path,
        default=DEFAULTS['PATHS']['REPO_ROOT'] / 'nklm' / 'data' / 'analyses',
        help='directory to save tokens to as CSV (default: %(default)s)'
    )
    return parser.parse_args()


def parse_articles(articles_csv_filepath: Path) -> pd.Series:
    """
    Parse CSV containing article raw texts, returning only article bodies.

    Parameters
    ----------
    articles_csv_filepath : Path
        path to CSV containing article raw texts

    Returns
    -------
    articles : pd.Series
        article bodies
    """

    return pd.read_csv(articles_csv_filepath, usecols=['body']).squeeze()


def write_output(lemmas: List[Token], output_directory: Path) -> None:
    """
    Write lemmas and tallies to CSV.

    Parameters
    ----------
    lemmas : List[Token]
        list of all lemmatized tokens across all essays, excluding stop words,
        whitespace, and punctuation
    output_directory : Path
        directory to save tokens to as CSV
    """

    output_directory.mkdir(exist_ok=True, parents=True)
    output_filepath = output_directory / 'unique_tokens.csv'

    (pd.Series(lemmas).value_counts(normalize=True)
                      .to_frame('pct')
                      .rename_axis('token')
                      .reset_index()
                      .sort_values(by='pct', ascending=False)
                      .to_csv(output_filepath, index=False))

    print('Written:', output_filepath)


def main() -> None:
    args = parse_args()

    nlp = spacy.load('en_core_web_lg')

    articles = parse_articles(args.articles_csv_filepath)
    lemmas = get_all_lemmas(articles, nlp)

    write_output(lemmas, args.output_directory)


if __name__ == '__main__':
    main()
