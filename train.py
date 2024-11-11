import os, sys
import typing as T
from itertools import islice
from pathlib import Path
from sys import stdout

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_and_preprocess_data, UDData
from configs import TransitionConfig, GraphConfig

from transition_model import ParserModel
from graph_model import Batch, GraphDepModel

@click.group()
@click.option('--debug', is_flag=True)
@click.pass_context
def main(ctx, debug, dir_path):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['path'] = dir_path

@main.command()
@click.pass_context
def transition(ctx):
    '''Train the Transition model'''

    debug = ctx.obj['debug']
    dir_path = ctx.obj['path']

    print(80 * '=')
    print(f'INITIALIZING{" debug mode" if debug else ""}')
    print(80 * '=')
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    config = TransitionConfig()

    data = load_and_preprocess_data(
        dir_path=dir_path,
        word_embedding_path=dir_path / 'word2vec.pkl.gz',
        max_batch_size=config.batch_size,
        transition_cache=0 if debug else None)

    transducer, word_embeddings, train_data = data[:3]
    dev_sents, dev_arcs = data[3:5]
    test_sents, test_arcs = data[5:]
    config.n_word_ids = len(transducer.id2word) + 1  # plus null
    config.n_tag_ids = len(transducer.id2tag) + 1
    config.n_deprel_ids = len(transducer.id2deprel) + 1
    config.embed_size = word_embeddings.shape[1]
    for (word_batch, tag_batch, deprel_batch), td_batch in \
            train_data.get_iterator(shuffled=False):
        config.n_word_features = word_batch.shape[-1]
        config.n_tag_features = tag_batch.shape[-1]
        config.n_deprel_features = deprel_batch.shape[-1]
        config.n_classes = td_batch.shape[-1]
        break
    print(f'# word features: {config.n_word_features}')
    print(f'# tag features: {config.n_tag_features}')
    print(f'# deprel features: {config.n_deprel_features}')
    print(f'# classes: {config.n_classes}')
    if debug:
        dev_sents = dev_sents[:500]
        dev_arcs = dev_arcs[:500]
        test_sents = test_sents[:500]
        test_arcs = test_arcs[:500]

    print(80 * '=')
    print('TRAINING')
    print(80 * '=')
    weights_file = Path('weights-transition.pt')
    print('Best weights will be saved to:', weights_file)
    model = ParserModel(transducer, config, word_embeddings)
    if torch.cuda.is_available():
        model = model.cuda()
    best_las = 0.
    trnbar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(desc='Training', total=config.n_epochs, leave=False,
              unit='epoch', position=0, bar_format=trnbar_fmt) as progbar:
        for epoch in range(config.n_epochs):
            if debug:
                trn_loss = model.fit_epoch(list(islice(train_data, 32)), epoch,
                                           progbar, config.batch_size)
            else:
                trn_loss = model.fit_epoch(train_data, epoch, progbar)
            tqdm.write(f'Epoch {epoch + 1:>2} training loss: {trn_loss:.3g}')
            stdout.flush()
            dev_las, dev_uas = model.evaluate(dev_sents, dev_arcs)
            best = dev_las > best_las
            if best:
                best_las = dev_las
                if not debug:
                    torch.save(model.state_dict(), str(weights_file))
            tqdm.write(f'         validation LAS: {dev_las:.1%}'
                       f'{" (BEST!)" if best else  "        "} '
                       f'UAS: {dev_uas:.1%}')
    if not debug:
        print()
        print(80 * '=')
        print('TESTING')
        print(80 * '=')
        print('Restoring the best model weights found on the dev set.')
        model.load_state_dict(torch.load(str(weights_file)))
        stdout.flush()
        las, uas = model.evaluate(test_sents, test_arcs)
        if las:
            print(f'Test LAS: {las:.1%}', end='       ')
        print(f'UAS: {uas:.1%}')
        print('Done.')
    return 0

def do_eval(batch_iter: T.Iterable[Batch], model: GraphDepModel,
            desc: T.Optional[str] = None) -> T.Tuple[float, float, float]:
    uas_correct, las_correct, total, tree_sent, tot_sent = 0, 0, 0, 0, 0
    with tqdm(batch_iter, desc=desc, leave=False, disable=None, unit='batch',
              dynamic_ncols=True) as it:
        for batch in it:
            b_ucorr, b_lcorr, b_tot, b_trees = model.eval_batch(batch)
            uas_correct += b_ucorr
            las_correct += b_lcorr
            total += b_tot
            tree_sent += b_trees
            tot_sent += len(batch[4])
            it.set_postfix(LAS=f'{100 * las_correct / total:.1f}')
    return uas_correct / total, las_correct / total, tree_sent / tot_sent

@main.command()
@click.pass_context
def graph(ctx):
    '''Train the Graph model'''

    debug = ctx.obj['debug']
    dir_path = ctx.obj['path']

    print(80 * '=')
    print(f'INITIALIZING{" debug mode" if debug else ""}')
    print(80 * '=')
    torch.manual_seed(1234)
    if gpu := torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    cfg = GraphConfig()
    cfg.data_dir = dir_path / 'corpora'
    cfg.model_dir = dir_path / 'transformers'

    print('Reading data...', end=' ', flush=True)
    train, dev, test = UDData.read(cfg.data_dir,
                                   *cfg.ud_corpus,
                                   fraction=0.1 if debug else 1)
    print('Done.')

    print('Initializing model...', end=' ', flush=True)
    model = GraphDepModel(cfg, len(train.deprels_s2i))
    if gpu:
        model = model.cuda()
    print('Done.', flush=True)

    print(80 * '=')
    print(f'TRAINING{" (debug mode)" if debug else ""}')
    print(80 * '=')
    train_dl = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0 if debug else 2,
                          collate_fn=model.collate, pin_memory=gpu,
                          persistent_workers=not debug)
    dev_dl = DataLoader(dev, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0 if debug else 2,
                        collate_fn=model.collate, pin_memory=gpu,
                        persistent_workers=not debug)
    weights_file = Path('weights-graph.pt')
    print(f'Best weights will be saved to {weights_file}.\n')

    barfmt = ('{l_bar}{bar}| %d/' + str(cfg.n_epochs)
              + ' [{elapsed}<{remaining}{postfix}]')
    epoch_steps = len(train_dl) + len(dev_dl)
    best_las = 0.

    with tqdm(total=cfg.n_epochs * epoch_steps, desc='Training', disable=None,
              unit='epoch', dynamic_ncols=True, bar_format=barfmt % 0) as pbar:
        for epoch in range(1, cfg.n_epochs + 1):
            with tqdm(train_dl, desc=f'Epoch {epoch}', leave=False,
                      disable=None, unit='batch', dynamic_ncols=True) as it:
                for batch in it:
                    trn_loss = model.train_batch(batch)
                    it.set_postfix(loss=trn_loss)
                    pbar.update()
            dev_uas, dev_las, dev_trees = do_eval(dev_dl, model, 'Validating')
            if best := dev_las > best_las:
                best_las = dev_las
                torch.save(model.state_dict(), str(weights_file))
            tqdm.write(f'Epoch {epoch:>2} validation LAS: {dev_las:.1%}'
                       f'{" (BEST!)" if best else  "        "} '
                       f'UAS: {dev_uas:.1%} Trees: {dev_trees:.1%}')
            pbar.bar_format = barfmt % epoch

    print(80 * '=')
    print('TESTING')
    print(80 * '=')
    print('Restoring the best model weights found on the dev set...',
          end=' ', flush=True)
    model.load_state_dict(torch.load(str(weights_file)))
    print('Done.', flush=True)

    tst_dl = DataLoader(test, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0 if debug else 2,
                        collate_fn=model.collate, pin_memory=gpu,
                        persistent_workers=not debug)
    tst_uas, tst_las, tst_trees = do_eval(tst_dl, model, 'Testing')
    print(f'Test LAS: {tst_las:.1%} UAS: {tst_uas:.1%} Trees: {tst_trees:.1%}')

    print(80 * '=')
    print('DONE')
    print(80 * '=')

if __name__ == '__main__':
    main()
