import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm.auto import tqdm

class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size, embed_size=None, num_layers=1):
    super(EncoderRNN, self).__init__()

    if embed_size is None:
      embed_size = hidden_size

    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
    self.gru = nn.GRU(embed_size, hidden_size, num_layers)

  def forward(self, input, input_lengths, hidden=None):
    batch_size = len(input_lengths)
    embedded = self.embedding(input).view(-1, batch_size, self.hidden_size)
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)#.to(input.device)
    output, hidden = self.gru(packed, hidden if hidden is not None else self.initHidden(batch_size).to(input.device))
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpack (back to padded)
    return output, hidden

  def initHidden(self, batch_size=1):
    return torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True)

class DecoderRNN(nn.Module):
  def __init__(self, output_size, hidden_size, embed_size=None, num_layers=1):
    super(DecoderRNN, self).__init__()

    if embed_size is None:
      embed_size = hidden_size

    self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
    self.gru = nn.GRU(embed_size, hidden_size, num_layers)
    self.out = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden, dummy1, dummy2):
    embedded = self.embedding(input).view(1,1,-1)
    embedded = F.relu(embedded)
    output, hidden = self.gru(embedded, hidden)
    output = self.out(output[0])
    output = self.softmax(output)
    return output, hidden, None, None

class Attention(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attention, self).__init__()

    self.method = method
    self.hidden_size = hidden_size

    if method == 'general':
      self.attn = nn.Linear(hidden_size, hidden_size)
    
    elif method == 'concat':
      self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
      self.other = nn.Parameter(torch.FloatTensor(hidden_size))

  def forward(self, queries, keys):
    n_keys, batch_size = keys.size()[:2]

    # query: 1 x bs x hs
    # keys: ks x bs x hs
    # scores: 1 x bs x hs
    
    scores = torch.zeros(batch_size, n_keys, device=queries.device)

    for b in range(batch_size):
      for i in range(n_keys):
        scores[b, i] = self.score(queries[0, b], keys[i, b])

    return F.softmax(scores, dim=1).unsqueeze(0)

  def score(self, query, key):

    if self.method == 'dot':
      score = query.dot(key)

    elif self.method == 'general':
      score = self.attn(key)
      score = query.dot(key)

    elif self.method == 'concat':
      score = self.attn(torch.cat((query, key), 0))
      score = self.other.dot(score)

    else:
      assert False, f'Unknown method: {self.method}'

    return score


class AttentionDecoderRNN(nn.Module):
  def __init__(self, attn, output_size, hidden_size, embed_size=None, num_layers=1, dropout_p=0.1):
    super(AttentionDecoderRNN, self).__init__()

    if embed_size is None:
      embed_size = hidden_size

    self.embed_size = embed_size
    self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
    self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers, dropout=dropout_p)
    self.out = nn.Linear(hidden_size * 2, output_size)

    self.attn = Attention('dot' if attn is None else attn, hidden_size)

  def forward(self, input, last_hidden, last_context, enc_outputs):

    # for k in ['input', 'last_hidden', 'last_context', 'enc_outputs']:
    #   print(k, locals()[k].device)

    batch_size = input.size(0) #enc_outputs.size(1)
    embedded = self.embedding(input).view(-1, batch_size, self.embed_size)

    rnn_input = torch.cat((embedded, last_context), 2)
    rnn_output, hidden = self.gru(rnn_input, last_hidden) # 1,B,H

    # for k in ['rnn_output', 'enc_outputs']:
    #   print(k, locals()[k].size())
      
    attn_weights = self.attn(rnn_output, enc_outputs)

    # for k in ['attn_weights', 'enc_outputs']:
    #   print(k, locals()[k].device)

    context = torch.bmm(attn_weights.transpose(0,1), enc_outputs.transpose(0,1)).transpose(0,1) # B,1,H

    # for k in ['attn_weights', 'context']:
    #   print(k, locals()[k].size())

    rnn_output = rnn_output.squeeze(0) # B,H
    context = context.squeeze(0) # B,H
    output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1) # B,OUT

    # for k in ['output']:
    #   print(k, locals()[k].size())

    return output, hidden, context.unsqueeze(0), attn_weights

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

import random
from pathlib import Path

MAX_LENGTH=256

class Seq2SeqAtt(nn.Module):

  def __init__(self, eos_token_id, encoder, decoder, max_length=MAX_LENGTH, device=None):
    super(Seq2SeqAtt, self).__init__()

    self.max_length = max_length
    self.device = device if device is not None else 'cpu'
    self.encoder = encoder
    self.decoder = decoder

    self.eos_token_id = eos_token_id

  def forward(self, inputs, max_length=MAX_LENGTH):
    output, attn_wgt, _ = self.process_batch(inputs, max_length=max_length)
    return output, attn_wgt

  def process_batch(self, inputs_batch, input_lengths=None,
                    targets_batch=None, target_lengths=None,
                    max_length=MAX_LENGTH,
                    teacher_forcing_rate=0.5,
                    criterion=None):

    device = self.device
    encoder = self.encoder.to(device)
    decoder = self.decoder.to(device)

    max_length = min(max_length, self.max_length)
    batch_size = inputs_batch.size(1)

    if input_lengths is None:
      input_lengths = [inputs_batch.size(0)] * batch_size

    target_exists = True if (targets_batch is not None) and (target_lengths is not None) else False

    assert inputs_batch.size(1) == len(input_lengths)

    inputs_batch = inputs_batch.to(device)
    if target_exists:
      n_targets = [min(target_length, max_length) for target_length in target_lengths]
      targets_batch = targets_batch.to(device)

    # run through encoder
    enc_output, enc_hidden = encoder(inputs_batch, input_lengths)

    # run decoder and collect outputs
    dec_outputs = torch.zeros(batch_size, max_length, device=device)
    dec_attentions = torch.zeros(max_length, batch_size, max_length, device=device)

    dec_input = torch.tensor([self.eos_token_id] * batch_size, device=device)
    dec_hidden = enc_hidden
    dec_context = torch.zeros(1, batch_size, encoder.hidden_size, device=device)

    loss, cnt = 0, 0
    for di in range(max(n_targets) if target_exists else max_length):
      dec_output, dec_hidden, dec_context, attn_weights = decoder(
        dec_input, dec_hidden, dec_context, enc_output
      )

      # print(f'di === {di}, dec_input === {dec_input}')
      # for k in ['dec_input', 'dec_output', 'dec_context', 'attn_weights','dec_attentions']:
      #   print(k, locals()[k].size())
  
      if attn_weights is not None:
        aw_length = min(attn_weights.size(-1), max_length)
        # print(attn_weights.data[0, :, :aw_length])
        dec_attentions[di, :, :aw_length] += attn_weights.data[0, :, :aw_length]

      dec_input_next = torch.zeros(batch_size, dtype=torch.long)

      for bi in range(len(dec_input)):
        if target_exists and (di < n_targets[bi]) and (dec_input[bi] > 0): # pad_id === 0
          # print('bi:', bi)
          # print('   ', dec_output[bi])
          # print('   ', targets_batch[di, bi])
          loss += criterion(dec_output[bi], targets_batch[di, bi])
          cnt += 1
          # print('    loss =', loss)
          # print('     cnt =', cnt)

        topv, topi = dec_output[bi].data.topk(1)
        dec_outputs[bi, di] = topi.item()

        if target_exists and (di < n_targets[bi]) and (random.random() < teacher_forcing_rate):
          dec_input_next[bi] = targets_batch[di, bi]
        else:
          # print('             ', topi)
          # print('             ', topi.item())
          # print('             ', topi.squeeze().detach())
          dec_input_next[bi] = topi.squeeze().detach()

      dec_input = dec_input_next.to(device)

      # if dec_input == self.eos_token_id:
      #   break
  
    return (
      dec_outputs[:, :di+1],
      dec_attentions[:di+1, :, :len(enc_output)].transpose(0, 1),
      loss #(loss/cnt if cnt > 0 else None)
    )


  def process_(self, inputs, targets=None, max_length=MAX_LENGTH, teacher_forcing_rate=0.5, criterion=None):
    device = self.device
    encoder = self.encoder.to(device)
    decoder = self.decoder.to(device)

    max_length = min(max_length, self.max_length)

    inputs = inputs.to(device)
    if targets is None:
      n_targets = 0
    else:
      targets = targets.to(device)
      n_targets = min(targets.size(0), max_length)

    # run through encoder
    enc_hidden = encoder.initHidden().to(device)
    enc_outputs, enc_hidden = encoder(inputs, enc_hidden)

    # run decoder and collect outputs
    dec_outputs = torch.zeros(max_length, device=device)
    dec_attentions = torch.zeros(max_length, max_length, device=device)

    dec_input = torch.tensor([[self.eos_token_id]], dtype=torch.long, device=device)
    dec_hidden = enc_hidden
    dec_context = torch.zeros(1, encoder.hidden_size, device=device)

    loss = 0
    for di in range(max_length):
      dec_output, dec_hidden, dec_context, dec_attention = decoder(
        dec_input, dec_hidden, dec_context, enc_outputs
      )

      # for k in ['dec_context', 'dec_attention']:
      #   print(k, locals()[k].size())
      # print('dec_attention.data', dec_attention.data.size())

      if dec_attention is not None:
        aw_length = min(dec_attention.size(2), max_length)
        dec_attentions[di, :aw_length] += dec_attention.data[0][0][:aw_length]

      if di < n_targets:
        loss += criterion(dec_output, targets[di])

      topv, topi = dec_output.data.topk(1)
      dec_outputs[di] = topi.item()

      if (di < n_targets) and (random.random() < teacher_forcing_rate):
        dec_input = targets[di]
      else:
        dec_input = topi.squeeze().detach()

      if dec_input == self.eos_token_id:
        break
  
    return dec_outputs[:di+1], dec_attentions[:di+1, :len(enc_outputs)], loss

from .utils import get_batch, timeSince, showPlot

class Seq2SeqHelper():

  def __init__(self, tokenizer, model, optimizer=None, lr=2e-5, name=None, workdir=None):
    self.__timestamp__()

    self.name = name if name is not None else f'noname-{self.timestamp}'
    self.workdir = Path('runs' if workdir is None else workdir)/self.name
    self.workdir.mkdir(parents=True, exist_ok=True)

    self.tokenizer = tokenizer
    self.model = model
    self.optimizer = optim.SGD(self.model.parameters(), lr=lr) if optimizer is None else optimizer

    self.teacher_forcing_rate = 0.5 #1.0
    self.clip = 5.0

  def __timestamp__(self, timestamp=None):
    self.timestamp = time.time() if timestamp is None else timestamp

  def save_ckpt(self, epoch, loss, path=None):
    if path is None:
      ckpt_dir = self.workdir/'ckpt'
      ckpt_dir.mkdir(parents=True, exist_ok=True)
      path = ckpt_dir/f'{self.timestamp}_e{epoch:03}.pt'
    
    torch.save({
      'epoch': epoch,
      'loss': loss,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'timestamp': time.time(),
    }, path)

  def load_ckpt(self, path):
    ckpt = torch.load(path)
    self.__timestamp__(ckpt['timestamp'])
    self.model.load_state_dict(ckpt['model_state_dict'])
    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch']

  def evaluate(self, sentence, max_length=MAX_LENGTH):
    # inputs = torch.tensor(self.tokenizer(sentence).input_ids, dtype=torch.long)
    # inputs = torch.tensor(self.tokenizer(normalize_line(sentence)).input_ids, dtype=torch.long)
    inputs = torch.tensor(self.tokenizer.encode(sentence).ids, dtype=torch.long)

    with torch.no_grad():
      dec_outputs, dec_attentions = self.model(inputs.view(-1,1), max_length=max_length)
    
    dec_outputs = dec_outputs.squeeze(0)
    dec_attentions =  dec_attentions.squeeze(0)

    # return self.tokenizer.decode(dec_outputs.to('cpu')), dec_attentions.to('cpu')
    return self.tokenizer.decode(list(dec_outputs.to('cpu').int())), dec_attentions.to('cpu')

  def evaluateRandomly(self, pairs, n=10):
    for _ in range(n):
      pair = random.choice(pairs)
      print('<', pair[0])
      print('=', pair[1])
      dec_sentence, attentions = self.evaluate(pair[0])
      print('>', dec_sentence)
      print('')

  def train_pair(self, input_tensor, target_tensor, criterion, max_length=MAX_LENGTH):
    self.__timestamp__()
    self.optimizer.zero_grad()

    self.model.train(True)
    dec_outputs, dec_attentions, loss = self.model.process(
      input_tensor, 
      target_tensor, 
      teacher_forcing_rate=self.teacher_forcing_rate,
      criterion=criterion,
      max_length=max_length
    )

    loss.backward()
    # with torch.autograd.set_detect_anomaly(True):
    #   loss.backward()

    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
    self.optimizer.step()

    return loss.item() / len(dec_outputs)

  def train_batch(self, batch, criterion, max_length=MAX_LENGTH, teacher_forcing_rate=None):
    self.__timestamp__()

    if teacher_forcing_rate is None:
      teacher_forcing_rate = self.teacher_forcing_rate

    self.model.train()
    input_tensor, input_lengths, target_tensor, target_lengths = batch

    dec_outputs, dec_attentions, loss = self.model.process_batch(
      input_tensor,
      input_lengths,
      target_tensor, 
      target_lengths,
      teacher_forcing_rate=teacher_forcing_rate,
      criterion=criterion,
      max_length=max_length
    )

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item() / sum(target_lengths)

  def evaluate_batch(self, batch, criterion, max_length=MAX_LENGTH):

    with torch.no_grad():
      input_tensor, input_lengths, target_tensor, target_lengths = batch

      _, _, loss = self.model.process_batch(
        input_tensor,
        input_lengths,
        target_tensor, 
        target_lengths,
        teacher_forcing_rate=0,
        criterion=criterion,
        max_length=max_length
      )

    return loss.item() / sum(target_lengths)    

  def train(self, pairs, valid_pairs=None, batch_size=32, criterion=None, n_epochs=3, max_length=MAX_LENGTH, print_every=5, plot_every=10, shuffle=True, resume=None, skip=0, save=True, pad_token='[PAD]'):

    epoch = 0
    n_pairs = len(pairs)
    n_steps = n_epochs * n_pairs
    indices = list(range(n_pairs))

    if resume is not None and type(resume) == str:
      print(f'load checkpoint: {resume}')
      epoch = self.load_ckpt(resume)
      skip += epoch * n_pairs

    progbar = tqdm(range(n_steps))
    start = time.time()
    plot_losses = []

    criterion = nn.NLLLoss() if criterion is None else criterion

    with SummaryWriter(self.workdir) as writer:

      for epoch in range(n_epochs):

        if skip >= n_pairs:
          skip -= n_pairs
          progbar.update(n_pairs)
          continue

        if shuffle:
          random.shuffle(indices)

        if skip > 0:
          remain = indices[skip:]
          progbar.update(skip)
          skip = 0
        else:
          remain = indices

        print_loss_total = 0 # reset every print_every
        plot_loss_total = 0 # reset every plot_every
        epoch_loss_total = 0
        pad_id = pad_id=self.tokenizer.token_to_id(pad_token)

        for iter, batch in enumerate(get_batch(pairs, pad_id=pad_id, batch_size=batch_size, indices=remain)):

          loss = self.train_batch(batch, criterion)
          print_loss_total += loss
          plot_loss_total += loss
          epoch_loss_total += loss

          current_batch_size = len(batch[-1])
          progbar.update(current_batch_size)

          # print(f'{Utils.timeSince(start, iter/n_pairs)} ({iter} {iter/n_pairs*100:>4}%) {loss:.6}')

          if (iter+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{timeSince(start, progbar.n/n_steps)} S:{progbar.n}/{n_steps} E:{progbar.n/n_pairs:.2}/{n_epochs} L:{print_loss_avg:.6}')
            writer.add_scalars("Loss", {"train": print_loss_avg}, progbar.n)

          if (iter+1) % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)

        if valid_pairs:
          valid_loss = []
          valid_progbar = tqdm(range(len(valid_pairs)))
          for batch in get_batch(valid_pairs, pad_id=pad_id, batch_size=batch_size):
            valid_loss.append(self.evaluate_batch(batch, criterion))
            valid_progbar.update(len(batch[-1]))
          valid_loss_avg = torch.mean(torch.tensor(valid_loss))
          writer.add_scalars('Loss', {'valid': valid_loss_avg}, progbar.n)
          print(f'{timeSince(start, progbar.n/n_steps)} S:{progbar.n}/{n_steps} E:{progbar.n/n_pairs:.2}/{n_epochs} L:{valid_loss_avg:.6} --VALIDATION--')

        if save:
          self.save_ckpt(epoch+1, loss)

    if len(plot_losses) > 0:
      showPlot(plot_losses)

    return plot_losses

