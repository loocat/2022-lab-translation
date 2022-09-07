import re
import random
import unicodedata
from pathlib import Path
from datasets import Dataset, DatasetDict

def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
  )

def normalize_line(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r'([!?.])', r' \1', s)
  s = s.replace('"', '').replace('``', '')
  if s[-1] != '.':
    s += '.'
  return s

class Bitext():
  def __init__(self, tokenizer, bitext_dir, lang_source, lang_target, max_len_source, max_len_target, bitext_prefix=None):
    self.tokenizer = tokenizer
    self.bitext_dir = bitext_dir
    self.lang_source = lang_source
    self.lang_target = lang_target
    self.max_len_source = max_len_source
    self.max_len_target = max_len_target
    self.bitext_prefix = 'bitext' if bitext_prefix is None else bitext_prefix

  def load_bitext(self, split, language):
    file = self.bitext_dir/f'{self.bitext_prefix}.{split}.{language}'
    lines = file.read_text().strip().split('\n')
    return [normalize_line(line) for line in lines if len(line) > 0]

  def to_dataset(self, split):
    sentence_source = self.load_bitext(split, self.lang_source)
    sentence_target = self.load_bitext(split, self.lang_target)
    model_inputs = self.tokenizer(sentence_source, max_length=self.max_len_source, truncation=True)
    with self.tokenizer.as_target_tokenizer():
      targets = self.tokenizer(sentence_target, max_length=self.max_len_target, truncation=True)

    model_inputs['labels'] = targets['input_ids']
    return Dataset.from_dict(model_inputs)

  def to_datasets(self):
    dd = DatasetDict()
    for split in ['train', 'valid']:
      dd[split] = self.to_dataset(split)
    return dd

import torch

def pad_seq(seq, max_len, pad_id):
  return [seq[i] if i < len(seq) else pad_id for i in range(max_len)]

def get_batch(pairs, pad_id, batch_size=32, indices=None, shuffle=True):
  if indices is None:
    indices = list(range(len(pairs)))
    if shuffle:
      random.shuffle(indices)

  for b in range(0, len(pairs), batch_size):
    # Zip into pairs, sort by length (descending)
    seq_pairs = [pairs[i] for i in indices[b: b+batch_size]]
    seq_pairs = sorted(seq_pairs, key=lambda p: len(p[0]), reverse=True)

    # Unzip
    input_seqs, target_seqs = zip(*seq_pairs)
  
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths), pad_id) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths), pad_id) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_tensor = torch.LongTensor(input_padded).transpose(0, 1)
    target_tensor = torch.LongTensor(target_padded).transpose(0, 1)

    yield input_tensor, input_lengths, target_tensor, target_lengths


import time
from time import gmtime, strftime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def timeSince(since, percent):
  lap = time.time() - since
  end = lap / percent
  res = end - lap
  lap = strftime('%H:%M:%S', gmtime(lap))
  res = strftime('%H:%M:%S', gmtime(res))
  return strftime(f'{lap}<{res}')

def tensorFromIndices(ids):
  return torch.tensor(ids, dtype=torch.long).view(-1,1)
  
def tensorFromPair(pair):
  input_tensor = tensorFromIndices(pair[0])
  target_tensor = tensorFromIndices(pair[1])
  return (input_tensor, target_tensor)

def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)

def showAttention(tokenizer, input_sentence, output_sentence, attentions):
  # setup figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
  fig.colorbar(cax)

  # setup axes
  # ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
  # ax.set_yticklabels([''] + output_sentence.split())
  ax.set_xticklabels([''] + tokenizer.encode(input_sentence).tokens + ['[EOS]'], rotation=90)
  # with tokenizer.as_target_tokenizer():
  #   ax.set_yticklabels([''] + tokenizer.encode(output_sentence).tokens)
  ax.set_yticklabels([''] + tokenizer.encode(output_sentence).tokens + ['[EOS]'])

  # show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()

def evaluateAndShowAttention(model, input_sentence):
  output_sentence, attentions = model.evaluate(input_sentence)
  print('input=', input_sentence)
  print('output=', output_sentence)
  showAttention(model.tokenizer, normalize_line(input_sentence), normalize_line(output_sentence), attentions)
