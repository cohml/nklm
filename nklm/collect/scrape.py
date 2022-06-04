import argparse
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from urllib.request import urlopen

from nklm.util.defaults import DEFAULTS
from nklm.util.utils import full_path


URL_BASE = 'http://www.rodong.rep.kp/en/index.php?strPageID=SF01_02_01&newsID={}-000{}'


def date(iso: str) -> datetime.date:
    """
    Wrangle passed date string into a datetime.date object.

    Paramters
    ---------
    iso : str
        ISO-formatted date string (i.e., "YYYY-MM-DD")

    Returns
    -------
    dt : datetime.date
        datetime object for passed date string
    """

    return datetime.fromisoformat(iso).date()


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
    title = body = ''

    while not title:
        title = p.pop(0).text.strip()
        if len(p) == 0:
            break
    else:
        body = ' '.join(pi.text.strip() for pi in p).strip().replace('\xa0', ' ')

    # the HTML source for a small minority of pages contains duplicate paragraphs;
    # these cannot easily be parsed automatically, so I decided to just throw them out;
    # after plotting the lengths as a histogram, 1,500 characters for titles and 50,000
    # characters for bodies seemed like reasonable cutoffs
    if len(title) > 1.5e3 or len(body) > 5e4:
        title = body = ''

    return (title.replace('\xa0', ' '),
            body.replace('\xa0', ' '))


def get_dates_and_article_urls(
    start_date: datetime.date,
    end_date: datetime.date
) -> List[Tuple[str, str]]:
    """
    Construct a list of URLs to all possible articles published online between the
    requested start and end dates.

    Parameters
    ----------
    start_date : datetime.date
        date to start scraping articles from
    end_date : datetime.date
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
        type=date,
        default='2018-01-02', # NB: first date from which `BASE_URL` returns a hit
        help='First date to scrape articles for (must be formatted as "YYYY-MM-DD"). '
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-e', '--end_date',
        type=date,
        default=datetime.now().date(),
        help='Last date to scrape articles for (must be formatted as "YYYY-MM-DD"). '
             '(default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output_directory',
        type=full_path,
        default=DEFAULTS['PATHS']['REPO_ROOT'] / 'nklm' / 'data' / 'raw',
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


def webpage_is_valid(
    webpage: BeautifulSoup
) -> bool:
    """
    Determine whether the webpage actually contains any valid content.

    Parameters
    ----------
    webpage : BeautifulSoup
        full HTML source for a single webpage

    Returns
    -------
    is_valid : bool
        ``True`` if the webpage contains valid content, else ``False``
    """

    h1 = webpage.find('h1')

    if h1 is None:
        is_valid = True
    elif h1.text.strip() in ('알수 없는 주소', ''):
        is_valid = False

    return is_valid


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

    output_basename = f'articles_{args.start_date}_{args.end_date}.csv'
    output_filepath = args.output_directory / output_basename
    args.output_directory.mkdir(exist_ok=True, parents=True)

    articles.to_csv(output_filepath, index=False)
    print('Written:', output_filepath)


def main() -> None:
    args = parse_args()

    if args.start_date > args.end_date:
        err = (
            f'The end date ({args.end_date}) must be later than '
            f'or equal to the start date ({args.start_date}).'
        )
        raise argparse.ArgumentTypeError(err)

    articles = []
    dates_and_urls = get_dates_and_article_urls(args.start_date, args.end_date)
    tqdm_kwargs = {
        'desc' : f'Scraping from {args.start_date} to {args.end_date}',
        'total' : len(dates_and_urls),
        'unit' : ' articles'
    }

    for date, url in tqdm(dates_and_urls, **tqdm_kwargs):
        webpage = parse_webpage(url)

        if not webpage_is_valid(webpage):
            continue

        title, body = extract_article_title_and_body(webpage)

        if all([title, body]):
            articles.append({'date' : date,
                             'url' : url,
                             'title' : title,
                             'body' : body})

    write_output(articles, args)


if __name__ == '__main__':
    main()
