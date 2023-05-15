import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)

class Rnn(nn.Module):

    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            raise NotImplementedError('cell_type {} is not implemented'.format(cell_type))

    def forward(self, x):
        """
        Inputs:
        x - - (batch_size, seq_length, input_dim)
        Outputs:
        h - - bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h


class Sp_norm_model(nn.Module):
    def __init__(self, args):
        super(Sp_norm_model, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)


        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)


    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits


    def train_one_step(self, inputs, masks):    #input x directly to predictor
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_rationale(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z
    def pred_forward_logit(self, inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits


    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

class GenEncNoShareModel(nn.Module):

    def __init__(self, args):
        super(GenEncNoShareModel, self).__init__()
        self.lay=args.lay
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.cls = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        if self.lay:
            cls_outputs = self.layernorm2(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def get_rationale(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z

    def detach_gen_pred(self, inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)


        z_pred_detach=torch.detach(z)
        ########## Classifier ##########
        cls_embedding = embedding * (z_pred_detach[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        if self.lay:
            cls_outputs = self.layernorm2(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def pred_forward_logit(self, inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        if self.lay:
            cls_outputs = self.layernorm2(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs=self.layernorm2(outputs)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits
    def train_skew(self,inputs,masks,labels):
        masks_ = masks.unsqueeze(-1)

        labels_ = labels.detach().unsqueeze(-1)  # batch*1
        pos = torch.ones_like(inputs)[:, :10] * labels_
        neg = -pos + 1
        skew_pad = torch.cat((pos, neg), dim=1)
        latter = torch.zeros_like(inputs)[:, 20:]

        masks_ = torch.cat((skew_pad, latter), dim=1).unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = self.layernorm2(outputs)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def g_skew(self, inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_logits=self.generator(embedding)
        soft_log = self._independent_soft_sampling(gen_logits)
        return soft_log

class GenEncShareModel(nn.Module):

    def __init__(self, args):
        super(GenEncShareModel, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.enc = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.hidden_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    # inputs (batch_size, seq_length)
    # masks (batch_size, seq_length)
    def forward(self, inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.enc(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = self.layernorm(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        # LSTM
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits









