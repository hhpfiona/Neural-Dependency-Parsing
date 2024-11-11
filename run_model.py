import os, sys
import click
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import typing as T
import xml.etree.ElementTree as ET
from operator import itemgetter
from pathlib import Path
from conllu import TokenList

from tqdm import tqdm

from data import UDData, load_and_preprocess_data
from configs import TransitionConfig, GraphConfig
from transition_parse import minibatch_parse
from transition_model import ParserModel

from transition_parse import minibatch_parse
from transition_model import ParserModel
from graph_model import GraphDepModel


@click.group()
@click.option('--dir-path', type=Path, default='/u/csc485h/fall/pub/a1/')
@click.pass_context
def main(ctx, dir_path):
    ctx.ensure_object(dict)
    ctx.obj['path'] = dir_path

@main.command()
@click.pass_context
@click.option('--model-weight-path', type=Path, default='weights-transition.pt')
@click.option('--dev-output-path', type=Path, default='dev.transition_out.conllu')
@click.option('--test-output-path', type=Path, default='test.transition_out.conllu')
def transition(ctx, model_weight_path, dev_output_path, test_output_path):
    '''Run Transition model.'''

    dir_path = ctx.obj['path']
    torch.manual_seed(1234)
    if gpu := torch.cuda.is_available():
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
        transition_cache=0)
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

    weights_file = Path(model_weight_path)
    print(f'Loading model weights from {weights_file}...', end=' ', flush=True)
    model = ParserModel(transducer, config, word_embeddings)
    model.load_state_dict(torch.load(str(weights_file),
                                     map_location='cuda' if gpu else 'cpu'))
    if gpu:
        model = model.cuda()
    model.eval()
    print('Done.', flush=True)
    dev_out, tst_out = Path(dev_output_path), Path(test_output_path)
    print(f'Outputs will be saved to {dev_out} and {tst_out}.\n')

    def run_and_save(sentences, filepath, desc):
        print(f'Running model on {desc.lower()}...')
        pred_deps = minibatch_parse(sentences, model, model.cfg.batch_size)
        pred_deps = [sorted(preds, key=itemgetter(1)) for preds in pred_deps]
        with filepath.open('w') as fout:
            for sent, pdeps in zip(sentences, pred_deps):
                if len(sent) == len(pdeps):
                    tl = TokenList(
                        [{'id': i, 'form': w, 'lemma': '_', 'upos': '_',
                          'xpos': '_', 'feats': '_', 'head': h, 'deprel': l,
                          'deps': '_', 'misc': '_'}
                         for (w, p), (h, i, l) in zip(sent, pdeps)])
                else:
                    tl = TokenList([{'id': i, 'form': w, 'lemma': '_',
                                     'upos': '_', 'xpos': '_', 'feats': '_',
                                     'head': '_', 'deprel': '_', 'deps': '_',
                                     'misc': '_'}
                                    for i, (w, p) in enumerate(sent, start=1)])
                    for h, i, l in pdeps:
                        i -= 1
                        tl[i]['head'], tl[i]['deprel'] = h, l
                fout.write(tl.serialize())

    run_and_save(dev_sents, dev_out, 'Dev set')
    run_and_save(test_sents, tst_out, 'Test set')

@main.command()
@click.pass_context
@click.option('--model-weight-path', type=Path, default='weights-graph.pt')
@click.option('--dev-output-path', type=Path, default='dev.graph_out.conllu')
@click.option('--test-output-path', type=Path, default='test.graph_out.conllu')
def graph(ctx, model_weight_path, dev_output_path, test_output_path):

    dir_path = ctx.obj['path']

    cfg = GraphConfig()
    cfg.data_dir = dir_path / 'corpora'
    cfg.model_dir = dir_path / 'transformers'

    if gpu := torch.cuda.is_available():
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        print('Running on CPU.')
    torch.manual_seed(0)

    print('Reading data...', end=' ', flush=True)
    data_root = cfg.data_dir / f'UD_{cfg.ud_corpus[0]}-{cfg.ud_corpus[1]}'
    deprel_s2i = {'root': 0}  # keep root deprel as 0 for simplicity
    deprel_i2s = ['root']
    for dep in ET.parse(data_root / 'stats.xml').getroot().iterfind('.//dep'):
        if (deprel := dep.attrib['name']) not in deprel_s2i:
            deprel_s2i[deprel] = len(deprel_s2i)
            deprel_i2s.append(deprel)
    dev = UDData(data_root / 'en_ewt-ud-dev.conllu', deprel_s2i)
    test = UDData(data_root / 'en_ewt-ud-test.conllu', deprel_s2i)
    print('Done.')

    weights_file = Path('weights-graph.pt')
    print(f'Loading model weights from {weights_file}...', end=' ', flush=True)
    model = GraphDepModel(cfg, len(dev.deprels_s2i))
    model.load_state_dict(torch.load(str(weights_file),
                                     map_location='cuda' if gpu else 'cpu'))
    # NOTE: there's a bug in PyTorch that causes a relatively harmless error
    #  message if pin_memory=True and num_workers>0. since pin_memory
    #  doesn't make a huge difference here, it's hard-set to False.
    #  The bug is fixed as of v1.9.1, so once that version is being used,
    #  we can revert to using pin_memory=gpu here.
    if gpu:
        model = model.cuda()
    model.eval()
    print('Done.', flush=True)

    dev_dl = DataLoader(dev, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=model.collate,
                        pin_memory=False, persistent_workers=True)
    tst_dl = DataLoader(test, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=model.collate,
                        pin_memory=False, persistent_workers=True)
    dev_out, tst_out = Path(dev_output_path), Path(test_output_path)
    print(f'Outputs will be saved to {dev_out} and {tst_out}.\n')

    barfmt = ('{l_bar}{bar}| %d/2 [{elapsed}<{remaining}{postfix}]')
    tot_steps = len(dev_dl) + len(tst_dl)
    with tqdm(total=tot_steps, desc='Running', disable=None, unit='epoch',
              dynamic_ncols=True, bar_format=barfmt % 1) as pbar:
        def run_and_save(dl: DataLoader, ds: UDData, filepath: Path,
                         desc: str = ''):
            with tqdm(dl, desc=desc, leave=False, disable=None, unit='batch',
                      dynamic_ncols=True) as it, filepath.open('w') as fout:
                for i, batch in enumerate(it):
                    i *= cfg.BATCH_SIZE
                    sentences = ds[i:i + cfg.BATCH_SIZE][0]
                    batch = model.transfer_batch(batch)
                    pred_heads, pred_labels, _, _ = model._predict_batch(batch)
                    for sent, pheads, plabels in zip(sentences,
                                                     pred_heads.cpu(),
                                                     pred_labels.cpu()):
                        tl = TokenList(
                            [{'id': j, 'form': w, 'lemma': '_', 'upos': '_',
                              'xpos': '_', 'feats': '_', 'head': h.item(),
                              'deprel': deprel_i2s[l.item()], 'deps': '_',
                              'misc': '_'}
                             for j, (w, h, l) in
                             enumerate(zip(sent, pheads, plabels), start=1)])
                        fout.write(tl.serialize())
                    pbar.update()

        run_and_save(dev_dl, dev, dev_out, 'Dev set')
        pbar.bar_format = barfmt % 2
        run_and_save(tst_dl, test, tst_out, 'Test set')


if __name__ == '__main__':
    main()