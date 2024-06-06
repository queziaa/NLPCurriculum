import logging
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from data_loader import Batch, create_dataloaders, get_level_sizes, load_ontology
from loss import LabelSmoothing, SimpleLossCompute
from model import make_model
from optimizer import DenseSparseAdam


class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_train_epoch(data_iter, model, loss_fn, optimizer, scheduler, accum_iter, size_epoch, train_state):
    """
    Train a single epoch
    :param data_iter: torch.utils.data.DataLoader: iterator over train batches
    :param model: model instance
    :param loss_fn: SimpleLossCompute (see loss.py): generate predictions and compute loss
    :param optimizer: DenseSparseAdam optimizer
    :param scheduler: learning rate scheduler (ExponentialLR)
    :param accum_iter: int; gradient accumulation steps
    :param size_epoch: dataset_size / batch_size
    :param train_state: struct for tracking training state (see above)
    :return: tuple: (average loss, train_state)
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_fn(out, batch.tgt_y, batch.ntokens)
        loss_node.backward()

        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens

        if i % 40 == 1:
            lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start
            logging.info(f"Epoch Done: {(100 * i / size_epoch):.2f} | "
                         f"Acc Steps: {n_accum:3d} | "
                         f"Loss: {(loss / batch.ntokens):6.2f} | "
                         f"T/Sec: {(tokens / elapsed):7.1f} | "
                         f"LR: {lr:6.1e} | ")
            tokens = 0

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def run_eval(data_iter, model, loss_fn):
    """
    Evaluate model on the validation set
    :param data_iter: torch.utils.data.DataLoader: iterator over validation batches
    :param model: model instance
    :param loss_fn: SimpleLossCompute (see loss.py): generate predictions and compute loss
    :return: tuple: average loss, prec@5, ndcg@5
    """
    start = time.time()

    total_loss = 0
    total_tokens = 0

    for batch in data_iter:
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_fn(out, batch.tgt_y, batch.ntokens, valid=True)

        total_loss += loss.item()
        total_tokens += batch.ntokens

    avg_loss = total_loss / total_tokens

    elapsed = time.time() - start

    logging.info(f"Result Eval | Time Elapsed : {elapsed:.3f} |  Loss: {avg_loss}")
    return avg_loss


def train_model(vocab_src, vocab_tgt, tokenizer_src, config):
    """
    :param vocab_src: instance of torchtext.vocab.Vocab, mapping between tokens and their ids
    :param vocab_tgt: instance of torchtext.vocab.Vocab, mapping between labels and their ids
    :param tokenizer_src: function for text tokenization
    :param config: config
    :return:
    """
    gpu = 0

    logging.info(f"Train worker process using GPU")
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]

    emb_src_init = np.load(config["Paths"]["embedding_src"])
    print(emb_src_init.shape)
    label2level = load_ontology(config["Paths"]["ontology"])  # {str: int}
    level_sizes = get_level_sizes(config["Prediction"].getint("max_level"), vocab_tgt, label2level)

    # create model instance
    model = make_model(vocab_src, vocab_tgt, config, emb_src_init=emb_src_init)
    model.cuda(gpu)
    model_name = config["Model"]["name"]

    # loss function
    criterion = LabelSmoothing(
        level_sizes=level_sizes,
        padding_idx=pad_idx,
        criterion=config["Train"]["loss"],
        smoothing=config["Train"].getfloat("smoothing")
    )
    criterion.cuda(gpu)
    loss_fn = SimpleLossCompute(criterion)

    # optimizer & scheduler
    optimizer = DenseSparseAdam(
        model.parameters(),
        lr=config["Train"].getfloat("base_lr"),
        betas=(config["Train"].getfloat("beta1"), config["Train"].getfloat("beta2")),
        eps=config["Train"].getfloat("eps"),
        weight_decay=config["Train"].getfloat("weight_decay"),
    )

    lr_scheduler = ExponentialLR(
        optimizer=optimizer,
        gamma=config["Train"].getfloat("gamma")
    )

    # X_dataloader is instance of torch.utils.data.DataLoader, yields batched (src, tgt) pairs, where
    # src: 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
    # tgt: 2D tensor of shape (batch_size, max_padding_tgt) with relevant labels ids
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        config,
    )

    epoch_size = len(train_dataloader.dataset) / config["DataLoader"].getint("batch_size_train")
    nepochs = config["Train"].getint("num_epochs")

    # simple struct for tracking training state
    train_state = TrainState()
    best_loss = float('inf')

    for epoch in range(nepochs):
        model.train()
        logging.info(f"[GPU{gpu}] Epoch {epoch} Training ====")
        _, train_state = run_train_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            loss_fn,
            optimizer,
            lr_scheduler,
            accum_iter=config["Train"].getint("accum_iter"),
            size_epoch=epoch_size,
            train_state=train_state,
        )

        # evaluate model on validation dataset
        logging.info(f"Epoch {epoch} Evaluation ====")
        model.eval()
        with torch.no_grad():
            valid_loss = run_eval(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model=model,
                loss_fn=loss_fn,
            )

        if valid_loss < best_loss:
            best_loss = valid_loss
            logging.info(f"New Best Score, saving ...")
            file_path = f"models/{model_name}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, file_path)

        torch.cuda.empty_cache()
