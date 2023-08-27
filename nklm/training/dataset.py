from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Any, Dict, List

from datasets import Dataset as HFDataset
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Span
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizer

from config import TrainingConfig


@dataclass(init=False)
class RodongSinmunDataset(Dataset):

    size: int = field(default=property(len))
    max_length: int

    def __init__(self, config: TrainingConfig):
        super().__init__()

        # read raw data
        self.df = pd.read_csv(config.data_csv_path).assign(split='train')

        # optionally sentence-tokenize articles with spacy
        if config.sentence_tokenize is True:
            self.df = self._sentence_tokenize()

        # optionally split into train and test partitions
        if config.do_eval:
            random_state = np.random.RandomState(config.seed)
            test_indices = np.argwhere(
                random_state.random(size=len(self.df)) < config.test_proportion
            )
            self.df.loc[test_indices.flat, 'split'] = 'test'
            if self.df.split.unique().size != 2:
                raise ValueError(
                    f'The dataset size ({len(self.df)}) is incompatible with '
                    f'a test set proportion of {config.test_proportion}.'
                )

        # write examples to output
        config.output_directory.mkdir(exist_ok=True, parents=True)
        self.df.to_csv(config.output_directory / 'data.csv', index=False)

        # initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.model_name_or_path
        )
        if config.max_length is None:
            config.max_length = self.tokenizer.model_max_length
        self.max_length = config.max_length

        # tokenize examples for model
        self.examples = {}
        partitions = ['train']
        if config.do_eval:
            partitions += ['test']
        for partition in partitions:
            records = (
                self.df.loc[self.df.split == partition].to_dict(orient='records')
            )
            self.examples[partition] = HFDataset.from_list(records).map(
                self._encode,
                remove_columns=self.df.columns.tolist(),
                batched=True,
                num_proc=cpu_count(),
                desc=f'Encoding {partition}',
            )

    def __getitem__(self, i: int):
        return self.examples[i]

    def _encode(self, batch: Dict[str, List]) -> Dict[str, List]:
        return self.tokenizer(
            batch['body'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True,
            #add_special_tokens=True,
            #return_attention_mask=True,
        )

    def __len__(self):
        return len(self.examples)

    def _sentence_tokenize(self) -> List[Span]:
        metadata_fields = ['date', 'url', 'title']
        for field in metadata_fields:
            Doc.set_extension(field, default=None)
        nlp = spacy.load('en_core_web_md')
        num_proc = cpu_count()
        docs = tqdm(
            nlp.pipe(
                self._tag_sentences_with_article_metadata(
                    nlp, self.df.to_dict(orient='records')
                ),
                n_process=num_proc,
                disable=['tagger', 'attribute_ruler', 'lemmatizer', 'ner']
            ),
            unit=' articles',
            total=len(self.df),
            desc=f'Sentence tokenization ({num_proc=})',
        )
        return pd.DataFrame(
            {
                'date': doc._.date,
                'url': doc._.url,
                'title': doc._.title,
                'body': sent.text,
            }
            for doc in docs for sent in doc.sents
        )

    def _tag_sentences_with_article_metadata(
        self,
        nlp,
        articles: Dict[str, Any],
    ) -> Doc:
        """
        For every sentence in the dataset, set the date, URL, and title of its
        parent article as an attribute, allowing the sentence to be mapped back
        the article if desired.
        """
        metadata_fields = ['date', 'url', 'title']
        for article in articles:
            doc = nlp.make_doc(article['body'])
            for field in metadata_fields:
                setattr(doc._, field, article[field])
            yield doc
