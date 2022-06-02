import argparse
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from urllib.request import urlopen


URL_BASE = 'http://www.rodong.rep.kp/en/index.php?strPageID=SF01_02_01&newsID={}-000{}'


def extract_article_title_and_body(
    webpage: BeautifulSoup
) -> Tuple[str, str]:
    """
    Extract title and body from the HTML source for a single article.

    Parameters
    ----------
    webpage : BeautifulSoup
        full HTML source for a single article

    Returns
    -------
    title : str
        article title
    body : str
        article body with all newlines removed
    """

    p = webpage.find_all('p')

    title = p[0].text.strip()
    body = ' '.join(pi.text.strip() for pi in p[1:])

    return title, body


def get_dates_and_article_urls(
    start_date: datetime,
    end_date: datetime
) -> List[Tuple[str, str]]:
    """
    Construct a list of URLs to all possible articles published online between the
    requested start and end dates.

    Parameters
    ----------
    start_date : datetime
        date to start scraping articles from
    end_date : datetime
        date to stop scraping articles on (inclusive)

    Returns
    -------
    dates_and_urls : List[Tuple[str, str]]
        list of possible article URLs and associated dates for each
    """

    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
    dates_and_urls = [(d, URL_BASE.format(d, i)) for d in dates for i in range(1, 10)]

    return dates_and_urls


def parse_args() -> argparse.Namespace:
    """Set command line arguments."""

    parser = argparse.ArgumentParser(
        description='Scrape all articles between two dates from the Rodong Sinmun '
                    'website, then save them to CSV with the dates in the filename.'
    )
    parser.add_argument(
        '-s', '--start_date',
        type=datetime.fromisoformat,
        default='2018-01-02', # NB: first date from which `BASE_URL` returns a hit
        help='First date to scrape articles for (must be formatted as "YYYY-MM-DD"). '
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-e', '--end_date',
        type=datetime.fromisoformat,
        default=datetime.now().date(),
        help='Last date to scrape articles for (must be formatted as "YYYY-MM-DD"). '
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output_directory',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data' / 'raw',
        help='Directory to write output CSV to. (default: %(default)s)'
    )

    return parser.parse_args()


def parse_webpage(
    url: str
) -> BeautifulSoup:
    """
    Read and parse HTML source from a webpage, given a URL string.

    Parameters
    ----------
    url : str
        webpage URL

    Returns
    -------
    webpage : BeautifulSoup
        full HTML source for a single webpage
    """

    with urlopen(url) as site:
        html = site.read().decode()
        webpage = BeautifulSoup(html, 'html.parser')

    return webpage


def webpage_contains_article(
    webpage: BeautifulSoup
) -> bool:
    """
    Determine whether a webpage contains an article available for scraping.

    Parameters
    ----------
    webpage : BeautifulSoup
        full HTML source for a single webpage

    Returns
    -------
    contains_article : bool
        ``True`` if the webpage contains an article, else ``False``
    """

    h1 = webpage.find('h1')

    if h1 is not None and h1.text == '알수 없는 주소':
        contains_article = False
    else:
        contains_article = True

    return contains_article


def write_output(
    articles: List[Dict[str, str]],
    args: argparse.Namespace
) -> None:
    """
    Write all scraped articles to a single CSV file with the specified start and end
    dates as part of the filename.

    Parameters
    ----------
    articles : List[Dict[str, str]]
        list of dicts, one per scraped article, with keys for date, URL, title and body
    args : argparse.Namespace
        command line arguments
    """

    articles = pd.DataFrame(articles)
    articles = articles.apply(lambda s: s.str.strip())

    output_basename = f'articles_{args.start_date.date()}_{args.end_date.date()}.csv'
    output_filepath = args.output_directory / output_basename
    args.output_directory.mkdir(exist_ok=True, parents=True)

    articles.to_csv(output_filepath, index=False)
    print('Written:', output_filepath)


def main() -> None:
    args = parse_args()

    if args.start_date > args.end_date:
        err = (
            f'The end date ({args.end_date.date()}) must be later than '
            f'or equal to the start date ({args.start_date.date()}).'
        )
        raise argparse.ArgumentTypeError(err)

    articles = []
    dates_and_urls = get_dates_and_article_urls(args.start_date, args.end_date)
    tqdm_kwargs = {
        'desc' : f'Scraping from {args.start_date.date()} to {args.end_date.date()}',
        'total' : len(dates_and_urls),
        'unit' : ' articles'
    }

    for date, url in tqdm(dates_and_urls, **tqdm_kwargs):
        webpage = parse_webpage(url)

        if not webpage_contains_article(webpage):
            continue

        title, body = extract_article_title_and_body(webpage)
        articles.append({'date' : date,
                         'url' : url,
                         'title' : title,
                         'body' : body})

    write_output(articles, args)


if __name__ == '__main__':
    main()