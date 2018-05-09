import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
from .normalize import LayerNorm
class SlowLSTM(nn.Module):

    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        
        # r to hidden weights
        self.w_ri = P(T(hidden_size, input_size))
        self.w_rf = P(T(hidden_size, input_size))
        self.w_ro = P(T(hidden_size, input_size))
        self.w_rc = P(T(hidden_size, input_size))
        
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, r, c = hidden
        h = h.view(h.size(1), -1)
        r = r.view(r.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + th.mm(r, self.w_ri) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + th.mm(r, self.w_rf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + th.mm(r, self.w_ro) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass

class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, r_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.r2h = nn.Linear(r_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, r, c = hidden
        h = h.view(h.size(1), -1)
        r = r.view(r.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h) + self.r2h(r)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)

class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, r_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', ln_preact=True, learnable=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size,
                                            hidden_size=hidden_size,
                                            r_size=r_size,
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4*hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4*hidden_size, learnable=learnable)
            self.ln_r2h = LayerNorm(4*hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, r, c = hidden
        h = h.view(h.size(1), -1)
        r = r.view(r.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        r2h = self.r2h(r)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
            r2h = self.ln_r2h(r2h)
        preact = i2h + h2h + r2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)