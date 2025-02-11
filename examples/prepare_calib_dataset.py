import os
import json
import random
import argparse
import dataclasses

from enum import StrEnum
from typing import Optional, Callable, Dict, Any

import datasets
import transformers

import yt.wrapper as yt


TEXT_KEY = 'text'


class SampleSelectionStrategy(StrEnum):
    FIRST = 'first'
    RANDOM = 'random'
    LONGEST = 'longest'


@dataclasses.dataclass
class CalibrationConfig:
    batch_size: int
    max_samples: int
    max_seq_length: int
    formatting_func: Optional[Callable] = None
    truncation: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationConfig':
        return cls(
            batch_size=d['batch_size'],
            max_samples=d['max_samples'],
            max_seq_length=d['max_seq_length'],
            formatting_func=d['formatting_func'],
            truncation=d['truncation']
        )
    
    def to_dict(self):
        result = dataclasses.asdict(self)
        if result['formatting_func'] is not None:
            result['formatting_func'] = '<cannot be printed>'
        return result


def create_formatting_func(formatting_func_config: Dict) -> Callable:
    def formatting_func(sample):
        text_parts = []
        for part in formatting_func_config['template']:
            if part['type'] == 'from_sample':
                if part['key'] not in sample:
                    raise ValueError(f'Unknown sample key. The actual are: {list(sample.keys())}')
                text_parts.append(sample[part['key']])
            elif part['type'] == 'text':
                text_parts.append(part['text'])
            else:
                raise ValueError(f'Unknwon text part type: {part["type"]}')
        text = ' '.join(text_parts) # TODO space?
        for old, new in formatting_func_config['replacement_rules'].items():
            text = text.replace(old, new)
        return text
    return formatting_func



def read_data(dataset_path: str) -> datasets.Dataset:
    yt_client = yt.YtClient(proxy=os.environ['YT_PROXY'], token=os.environ['YT_TOKEN'])
    return datasets.Dataset.from_list(list(yt_client.read_table(dataset_path)))


def load_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def format(
    dataset: datasets.Dataset,
    text_key: str,
    formatting_func: Callable
) -> datasets.Dataset:
    def fn(sample):
        sample[text_key] = formatting_func(sample)
        return sample
    
    return dataset.map(fn, batched=False)


def tokenize(
    dataset: datasets.Dataset,
    text_key: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    truncation: bool,
    max_length: Optional[int] = None
) -> datasets.Dataset:
    def fn(sample):
        sample['input_ids'] = tokenizer(
            sample[text_key],
            truncation=truncation,
            max_length=max_length
        ).input_ids
        return sample

    return dataset.map(fn, batched=False)


# read table from yt or somewhere else preprocess and put it inside a directory for the next script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--calib-config', type=str, required=True)
    parser.add_argument('--sample-selection-strategy', type=SampleSelectionStrategy, required=True)
    parser.add_argument('--tokenizer-name-or-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    args = parser.parse_args()

    dataset = read_data(args.dataset)
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)

    with open(args.calib_config) as f:
        config = json.load(f)
    if 'formatting_func' in config:
        config['formatting_func'] = create_formatting_func(config['formatting_func'])
    config = CalibrationConfig.from_dict(config)

    assert args.sample_selection_strategy != SampleSelectionStrategy.LONGEST or not config.truncation

    dataset = format(dataset, TEXT_KEY, config.formatting_func)
    dataset = tokenize(dataset, TEXT_KEY, tokenizer, config.truncation, config.max_seq_length)

    dataset = dataset.filter(lambda x: len(x['input_ids']) <= config.max_seq_length)

    if args.sample_selection_strategy == SampleSelectionStrategy.FIRST:
        calib_dataset = dataset[:config.max_samples]
    elif args.sample_selection_strategy == SampleSelectionStrategy.RANDOM:
        indices = list(range(0, len(dataset)))
        if config.max_samples < len(dataset):
            indices = random.sample(range(0, len(dataset)), k=config.max_seq_length)
        calib_dataset = dataset.select(indices)
    elif args.sample_selection_strategy == SampleSelectionStrategy.LONGEST:
        dataset = sorted(dataset, key=lambda x: len(x['input_ids']), reverse=True)
        calib_dataset = datasets.Dataset.from_list(dataset[:config.max_samples])

    to_json_kwargs = {
        'force_ascii': False
    }
    calib_dataset.to_json(os.path.join(args.save_dir, 'calibration.json'), **to_json_kwargs)
