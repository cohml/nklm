# Overview

This repository contains data, source code, and domain-adapted language models for
generating fake North Korean propaganda using machine learning.


# Installation

Use of this repository requires a conda environment. To install it, run the following
script:

```bash
bash ./create_env.sh
```

You can activate the environment using the following command:

```bash
conda activate ./env
```


# Models

*TBD*


# Data

## Distributions

The data set used to fine-tune the pretrained language models consists of 9,800
English newspaper articles published to [Rodong Sinmun](http://www.rodong.rep.kp/en/)
between January 2, 2018 and June 3, 2022.

The raw data consists of the following columns:

|Header   |Dtype  |Description                                   |
|:-------:|:-----:|:---------------------------------------------|
|``date`` |``str``|publish date in ISO format                    |
|``url``  |``str``|URL used to scrape article                    |
|``title``|``str``|verbatim article title                        |
|``body`` |``str``|verbatim article content with newlines removed|

The lengths of the articles are distributed as follows:

|    | N Words|N Characters|
|:--:|-------:|-----------:|
|mean|  261.78|    1,637.95|
| sd |  324.69|    2,042.54|


## Scrape new data

To scrape new articles, first activate the environment.

```bash
conda activate ./env
```

Then run the following command, substituting in actual dates as appropriate:

```bash
scrape --start_date {YYYY-MM-DD} --end_date {YYYY-MM-DD}
```

If either of ``--start_date`` or ``--end_date`` are unspecified, the following default
values will be used for the missing argument:

* ``--start_date`` : 2018-01-02
* ``--end_date``   : {today's date}

To see all options for customizing your new data collection, run the following command:

```bash
scrape --help
```
