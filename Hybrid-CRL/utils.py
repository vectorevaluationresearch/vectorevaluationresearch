import json
import shutil
import logging
import argparse
from pathlib import Path

from gensim.models import Word2Vec

from tokenizers import ByteLevelBPETokenizer

from transformers import RobertaTokenizer, RobertaTokenizerFast


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_static_tokens():
    with open('static_tokens.txt', 'r') as file_obj:
        tokens = file_obj.readlines()

    tokens = [token.strip() for token in tokens]
    return tokens


def load_w2v_model(task):
    model_path = Path('outputs') / f"{task}_w2v.model"

    try:
        model = Word2Vec.load(str(model_path))
    except FileNotFoundError:
        logger.info('Word2Vec model not found.')
        train_w2v_model(task)
        model = Word2Vec.load(str(model_path))
    return model


def train_w2v_model(task):
    data_path = Path('datasets') / f"{task}_train.txt"
    with open(str(data_path), 'r') as file_obj:
        rows = file_obj.readlines()

    code_snippets = [row.split('\t')[1].split() for row in rows]
    logger.info(f"Training Word2Vec model for {task}-level bug detection.")
    model = Word2Vec(sentences=code_snippets, vector_size=768)
    logger.info(f"Training Word2Vec model complete.")

    model_path = Path('outputs')
    model_path.mkdir(exist_ok=True, parents=True)
    model_path = model_path / f"{task}_w2v.model"
    model.save(str(model_path))
    logger.info(f"Saved Word2Vec model in /outputs.")


def train_hybrid_tokenizer(task):
    data_path = Path('datasets') / f"{task}_train.txt"
    with open(str(data_path), 'r') as file_obj:
        rows = file_obj.readlines()

    code_snippets = [row.split('\t')[1] for row in rows]

    # BPE tokenizer accepts inputs as a set of files.
    tmp_data_path = Path('tmp')
    tmp_data_path.mkdir(exist_ok=True, parents=True)

    files = []
    for _id, code in enumerate(code_snippets):
        tmp_file_path = str(tmp_data_path / f'tmp{_id}.txt')
        with open(tmp_file_path, 'w') as file_obj:
            file_obj.write(code)
        files.append(tmp_file_path)

    path_to_tok = Path('tokenizer')
    path_to_tok.mkdir(exist_ok=True, parents=True)

    logger.info(f"Training BPE tokenizer for {task}-level bug detection.")
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + get_static_tokens()
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files,
                    vocab_size=50265,
                    min_frequency=2,
                    special_tokens=special_tokens)
    tokenizer.save_model(str(path_to_tok))
    shutil.rmtree(str(tmp_data_path))
    logger.info(f"Saved hybrid tokenizer in /tokenizer.")


def get_static_ids(w2v_model, tokenizer):
    with open(str(Path('tokenizer') / 'vocab.json'), 'r') as file_obj:
        static_token_dict = json.load(file_obj)

    static_mapper = {}
    static_tokens = get_static_tokens()
    vocab = set(w2v_model.wv.key_to_index)
    static_tokens = set(static_tokens).intersection(vocab)

    for tok in static_tokens:
        static_mapper[static_token_dict[tok]] = tok

    return static_mapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_w2v", action='store_true',
                        help="Train Word2Vec model")
    parser.add_argument("--train_tok", action='store_true',
                        help="Train byte-level BPE tokenizer")
    parser.add_argument("--task", type=str, required=True,
                        choices=['method', 'stmt'], help="Bug detection task.")
    args = parser.parse_args()

    if args.train_w2v:
        train_w2v_model(args.task)

    if args.train_tok:
        train_hybrid_tokenizer(args.task)
