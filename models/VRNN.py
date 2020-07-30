import torch
import numpy as np

class VRNN(torch.nn.Module):
  """
  PyTorch implementation of a many-to-many vanilla RNN that accepts
  variable length inputs of the form (batch, time, data). Also returns
  tensor of hidden unit activations over time.
  """
  def __init__(self, input_dim, hidden_dim, output_dim=None, bias=False, activation=torch.tanh):
    super(VRNN, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    if not output_dim:
      output_dim = input_dim
    self.output_dim = output_dim
    self.bias = bias
    self.activation = activation
    self.init_params()

  def init_params(self):
    input_dim = self.input_dim
    hidden_dim = self.hidden_dim
    output_dim = self.output_dim
    bias = self.bias

    hscale = 0.1  # scale for h0
    ifactor = 1.0 / np.sqrt(input_dim)
    hfactor = 1.0 / np.sqrt(hidden_dim)
    pfactor = 1.0 / np.sqrt(hidden_dim)

    # NOTE: weights are oriented so as to be left multiplied by a batch of inputs
    self.h0 = torch.nn.parameter.Parameter(torch.randn(hidden_dim) * hscale)
    self.wI = torch.nn.parameter.Parameter(torch.randn(input_dim, hidden_dim) * ifactor)
    self.wR = torch.nn.parameter.Parameter(torch.randn(hidden_dim, hidden_dim) * hfactor)
    self.wO = torch.nn.parameter.Parameter(torch.randn(hidden_dim, output_dim) * pfactor)
    
    self.bR = self.bO = 0
    if bias:
        self.bR = torch.nn.parameter.Parameter(torch.zeros(hidden_dim))
        self.bO = torch.nn.parameter.Parameter(torch.zeros(output_dim))

  def forward(self, input_seq, h0=None):
    def affine(x, w, b=0):
      return torch.matmul(x, w) + b

    def step(x, h):
      h = self.activation(affine(x, self.wI) + affine(h, self.wR, self.bR))
      x = affine(h, self.wO, self.bO)
      return x, h

    batch_size, T, input_dim = input_seq.shape
    h = h0 if h0 != None else self.h0
    O = []
    H = []
    for t in range(T):
      x = input_seq[:, t, :]
      o, h = step(x, h)
      O.append(o)
      H.append(h)
    return torch.stack(O, dim=1), torch.stack(H, dim=1)

class VRNNembed(torch.nn.Module):
	"""
	Convenience class. VRNN that learns h0 embedding.
	"""
	def __init__(self, start_dim=2, *args, **kwargs):
		super(VRNNembed, self).__init__()
		self.VRNN = VRNN(*args, **kwargs)
		self.start_dim = start_dim
		self.embed = torch.nn.Linear(start_dim, self.VRNN.hidden_dim)
		

	def forward(self, input_seq, start_state):
    # start state must be of shape (batch_size, start_dim) (i.e. 2D)
		h0 = self.embed(start_state)
		return self.VRNN(input_seq, h0)
