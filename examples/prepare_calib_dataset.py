import os
import os.path
import json
import random
import argparse

from enum import StrEnum
from typing import Optional, Callable, Dict, Any

import yaml
import jinja2
import datasets
import transformers
import numpy as np

import yt.wrapper as yt


TEXT_KEY = 'text'


class SampleSelectionStrategy(StrEnum):
    FIRST = 'first'
    RANDOM = 'random'
    LONGEST = 'longest'


def default_formatting_func(sample):
    return sample[TEXT_KEY]


def create_formatting_func(config: Dict) -> Callable:
    if 'template' in config:
        env = jinja2.Environment()
        template = env.from_string(config['template'])
    elif 'template_path' in config:
        dirname = os.path.dirname(config['template_path'])
        template_name = os.path.basename(config['template_path'])
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(dirname)
        )
        template = env.get_template(template_name)

    def formatting_func(sample):
        text = template.render(ctx=sample)
        for old, new in formatting_func_config['replacement_rules'].items():
            text = text.replace(old, new)
        return text
    return formatting_func


def read_data(dataset_path: str) -> datasets.Dataset:
    yt_client = yt.YtClient(proxy=os.environ['YT_PROXY'], token=os.environ['YT_TOKEN'])
    x = list(yt_client.read_table(dataset_path))
    random.shuffle(x)
    return datasets.Dataset.from_list(x)


def load_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def format(
    dataset: datasets.Dataset,
    text_key: str,
    formatting_func: Callable
) -> datasets.Dataset:
    def fn(sample):
        return {text_key: formatting_func(sample)}
    
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


def summary(dataset: datasets.Dataset) -> Dict[str, Any]:
    QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]

    num_tokens = [len(d) for d in dataset['input_ids']]
    num_tokens = np.array(num_tokens)

    result = {
        'num samples': len(num_tokens),
        'min': int(num_tokens.min().item()),
        'max': int(num_tokens.max().item()),
        'mean': float(num_tokens.mean().item())
    }

    for value, q in zip(np.quantile(num_tokens, QUANTILES), QUANTILES):
        result[f'q={q:.2f}'] = int(value)

    return result


# read table from yt or somewhere else preprocess and put it inside a directory for the next script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-seq-length', type=int, required=True)
    parser.add_argument('--truncation', action='store_true', default=False)
    parser.add_argument('--format', type=str, default=None)
    parser.add_argument('--sample-selection-strategy', type=SampleSelectionStrategy, required=True)
    parser.add_argument('--tokenizer-name-or-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = read_data(args.dataset)
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)

    formatting_func = default_formatting_func
    if args.format is not None:
        with open(args.format) as f:
            formatting_func_config = yaml.safe_load(f)
        formatting_func = create_formatting_func(formatting_func_config)

    dataset = format(dataset, TEXT_KEY, formatting_func)
    dataset = tokenize(dataset, TEXT_KEY, tokenizer, args.truncation, args.max_seq_length)

    print('Full dataset stats:')
    print(f'{json.dumps(summary(dataset), ensure_ascii=False, indent=4)}\n')

    dataset = dataset.filter(lambda x: len(x['input_ids']) <= args.max_seq_length)

    if args.max_samples is not None:
        if args.sample_selection_strategy == SampleSelectionStrategy.FIRST:
            calib_dataset = dataset.select(range(0, min(args.max_samples, len(dataset))))
        elif args.sample_selection_strategy == SampleSelectionStrategy.RANDOM:
            indices = list(range(0, len(dataset)))
            if args.max_samples < len(dataset):
                indices = random.sample(range(0, len(dataset)), k=args.max_samples)
            calib_dataset = dataset.select(indices)
        elif args.sample_selection_strategy == SampleSelectionStrategy.LONGEST:
            dataset = sorted(dataset, key=lambda x: len(x['input_ids']), reverse=True)
            calib_dataset = datasets.Dataset.from_list(dataset[:args.max_samples])
    else:
        calib_dataset = dataset

    print('Calibration dataset stats:')
    print(f'{json.dumps(summary(calib_dataset), ensure_ascii=False, indent=4)}')

    to_json_kwargs = {
        'force_ascii': False
    }
    calib_dataset.to_json(os.path.join(args.save_dir, 'calibration.json'), **to_json_kwargs)
