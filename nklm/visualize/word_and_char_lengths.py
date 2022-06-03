import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import FuncFormatter
from pathlib import Path


def compute_lengths(articles: pd.Series) -> pd.DataFrame:
    """
    Compute the number of words and characters in each article.

    Note: The character lengths include punctuation and whitespace.

    Parameters
    ----------
    articles : pd.Series
        article bodies

    Returns
    -------
    lengths : pd.DataFrame
        number of words and characters in each article
    """

    nwords = articles.str.split().str.len().rename('nwords')
    nchars = articles.str.len().rename('nchars')
    return pd.concat([nwords, nchars], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot the number of words and characters per article as a pair of '
                    'histograms, saving the result as PNG.'
    )
    parser.add_argument(
        '-a', '--articles_csv_filepath',
        type=lambda p: Path(p).resolve(),
        required=True,
        help='path to CSV containing article raw texts'
    )
    parser.add_argument(
        '-o', '--output_directory',
        type=lambda p: Path(p).resolve(),
        default=Path(__file__).resolve().parent.parent / 'plots',
        help='directory to save plot to as PNG (default: %(default)s)'
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


def plot(lengths: pd.DataFrame, output_directory: Path) -> None:
    """
    Plot the distributions of word and character lengths as a pair of histograms,
    then save to PNG.

    Parameters
    ----------
    lengths : pd.DataFrame
        number of words and characters in each article
    output_directory : Path
        directory to save plot to as PNG
    """

    alpha = 2/3
    color = 'silver'
    thousands_separator = lambda x, p: format(int(x), ',')
    major_xtick_formatter = FuncFormatter(thousands_separator)
    bin_sizes = {
        'nwords' : 100,
        'nchars' : 1000
    }
    titles = {
        'nwords' : 'Number of words per article',
        'nchars' : 'Number of characters per article (incl. whitespace and punctuation)'
    }
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)

    for column, ax in zip(lengths.columns, axes.flat):
        bins = np.arange(
            start=0,
            stop=lengths[column].max() + 100,
            step=bin_sizes[column]
        )
        lengths[column].plot(
            kind='hist',
            color=color,
            bins=bins,
            title=titles[column],
            alpha=alpha,
            logy=True,
            ax=ax
        )
        mu, sigma = lengths[column].agg(['mean', 'std'])
        bbox = dict(boxstyle='round', facecolor=color, alpha=alpha)
        stats_str = '\n'.join([fr'$\mu={mu:,.2f}$', fr'$\sigma={sigma:,.2f}$'])
        ax.text(0.99, 0.95, stats_str, transform=ax.transAxes, ha='right', va='top', bbox=bbox)
        ax.grid(axis='y', linestyle='--')
        ax.set_xticks(bins)
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.set_xlim(0, bins[-1] + bin_sizes[column])
        ax.get_xaxis().set_major_formatter(major_xtick_formatter)

    output_directory.mkdir(exist_ok=True, parents=True)
    output_filename = output_directory / 'word_and_char_lengths.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print('Written:', output_filename)


def main() -> None:
    args = parse_args()
    articles = parse_articles(args.articles_csv_filepath)
    lengths = compute_lengths(articles)
    plot(lengths, args.output_directory)


if __name__ == '__main__':
    main()
