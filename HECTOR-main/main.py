import argparse
import configparser
import logging
import os

from data_loader import load_vocab, load_tokenizer, load_ontology, build_init_embeddings
from logger import set_logger
from train import train_model


def print_config(config):
    config_str = []
    for sec in config.sections():
        config_str.append(f"{sec}:")
        for k, v in config.items(sec):
            config_str.append(f"\t{k}: {v}")
    logging.info("Config:\n%s" % "\n".join(config_str))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True, help="Used as a prefix for model name and log file name")
    parser.add_argument("--config", default="configs/config_MAG.ini")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists("models"):
        os.mkdir("models")

    set_logger(args)

    config.set("Model", "name", args.name)

    print_config(config)

    # Load spaCy tokenizer for English language
    spacy_en = load_tokenizer()

    # Tokenizer for source sequence (text in natural language)
    tokenizer_src = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]

    # Build <label: ontology level> mapping
    label2level = load_ontology(config["Paths"]["ontology"])

    # Each vocab is instance of torchtext.vocab.Vocab, which maps tokens/labels to their unique ids
    vocab_src, vocab_tgt = load_vocab(config, tokenizer_src, label2level)

    # Initialize token embeddings from GloVe
    build_init_embeddings(config, vocab_src)

    train_model(vocab_src, vocab_tgt, tokenizer_src, config)


if __name__ == '__main__':
    main()
