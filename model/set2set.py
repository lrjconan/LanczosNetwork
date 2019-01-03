import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Set2Set', 'Set2Vec']


class Set2SetLSTM(nn.Module):

  def __init__(self, hidden_dim):
    """ Implementation of customized LSTM for set2set """
    super(Set2SetLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.forget_gate = nn.Sequential(
        *[nn.Linear(2 * self.hidden_dim, self.hidden_dim),
          nn.Sigmoid()])
    self.input_gate = nn.Sequential(
        *[nn.Linear(2 * self.hidden_dim, self.hidden_dim),
          nn.Sigmoid()])
    self.output_gate = nn.Sequential(
        *[nn.Linear(2 * self.hidden_dim, self.hidden_dim),
          nn.Sigmoid()])
    self.memory_gate = nn.Sequential(
        *[nn.Linear(2 * self.hidden_dim, self.hidden_dim),
          nn.Tanh()])

    self._init_param()

  def _init_param(self):
    for m in [
        self.forget_gate, self.input_gate, self.output_gate, self.memory_gate
    ]:
      for mm in m:
        if isinstance(mm, nn.Linear):
          nn.init.xavier_uniform_(mm.weight.data)
          if mm.bias is not None:
            mm.bias.data.zero_()

  def forward(self, hidden, memory):
    """
      Args:
        hidden: shape N X 2D
        memory: shape N X D

      Returns:
        hidden: shape N X D
        memory: shape N X D
    """
    ft = self.forget_gate(hidden)
    it = self.input_gate(hidden)
    ot = self.output_gate(hidden)
    ct = self.memory_gate(hidden)

    memory = ft * memory + it * ct
    hidden = ot * torch.tanh(memory)

    return hidden, memory


class Set2Vec(nn.Module):

  def __init__(self, element_dim, num_step_encoder):
    """ Implementation of Set2Vec """
    super(Set2Vec, self).__init__()
    self.element_dim = element_dim
    self.num_step_encoder = num_step_encoder
    self.LSTM = Set2SetLSTM(element_dim)
    self.W_1 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
    self.W_2 = nn.Parameter(torch.ones(self.element_dim, 1))
    self.register_parameter('W_1', self.W_1)
    self.register_parameter('W_2', self.W_2)

    self._init_param()

  def _init_param(self):
    nn.init.xavier_uniform_(self.W_1.data)
    nn.init.xavier_uniform_(self.W_2.data)

  def forward(self, input_set):
    """
      Args:
        input_set: shape N X D

      Returns:
        output_vec: shape 1 X 2D
    """
    num_element = input_set.shape[0]
    element_dim = input_set.shape[1]
    assert element_dim == self.element_dim
    hidden = torch.zeros(1, 2 * self.element_dim).to(input_set.device)
    memory = torch.zeros(1, self.element_dim).to(input_set.device)

    for tt in range(self.num_step_encoder):
      hidden, memory = self.LSTM(hidden, memory)
      energy = torch.tanh(torch.mm(hidden, self.W_1) + input_set).mm(self.W_2)
      att_weight = F.softmax(energy, dim=0)
      read = (input_set * att_weight).sum(dim=0, keepdim=True)
      hidden = torch.cat([hidden, read], dim=1)

    return hidden


class Set2Set(nn.Module):

  def __init__(self, element_dim, num_step_encoder):
    """ Implementation of Set2Set """
    super(Set2Set, self).__init__()
    self.element_dim = element_dim
    self.num_step_encoder = num_step_encoder
    self.LSTM_encoder = Set2SetLSTM(element_dim)
    self.LSTM_decoder = Set2SetLSTM(element_dim)
    self.W_1 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
    self.W_2 = nn.Parameter(torch.ones(self.element_dim, 1))
    self.W_3 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
    self.W_4 = nn.Parameter(torch.ones(self.element_dim, 1))
    self.W_5 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
    self.W_6 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
    self.W_7 = nn.Parameter(torch.ones(self.element_dim, 1))
    self.register_parameter('W_1', self.W_1)
    self.register_parameter('W_2', self.W_2)
    self.register_parameter('W_3', self.W_3)
    self.register_parameter('W_4', self.W_4)
    self.register_parameter('W_5', self.W_5)
    self.register_parameter('W_6', self.W_6)
    self.register_parameter('W_7', self.W_7)

    self._init_param()

  def _init_param(self):
    for xx in [
        self.W_1, self.W_2, self.W_3, self.W_4, self.W_5, self.W_6, self.W_7
    ]:
      nn.init.xavier_uniform_(xx.data)

  def forward(self, input_set):
    """
      Args:
        input_set: shape N X D

      Returns:
        output_set: shape N X 1
    """
    num_element = input_set.shape[0]
    element_dim = input_set.shape[1]
    assert element_dim == self.element_dim
    hidden = torch.zeros(1, 2 * self.element_dim).to(input_set.device)
    memory = torch.zeros(1, self.element_dim).to(input_set.device)

    # encoding
    for tt in range(self.num_step_encoder):
      hidden, memory = self.LSTM_encoder(hidden, memory)
      energy = torch.tanh(torch.mm(hidden, self.W_1) + input_set).mm(self.W_2)
      att_weight = F.softmax(energy, dim=0)
      read = (input_set * att_weight).sum(dim=0, keepdim=True)
      hidden = torch.cat([hidden, read], dim=1)

    # decoding
    memory = torch.zeros_like(memory)
    output_set = []
    for tt in range(num_element):
      hidden, memory = self.LSTM_decoder(hidden, memory)
      energy = torch.tanh(torch.mm(hidden, self.W_3) + input_set).mm(self.W_4)
      att_weight = F.softmax(energy, dim=0)
      read = (input_set * att_weight).sum(dim=0, keepdim=True)
      hidden = torch.cat([hidden, read], dim=1)
      energy = torch.tanh(torch.mm(read, self.W_5) + torch.mm(
          input_set, self.W_6)).mm(self.W_7)
      output_set += [torch.argmax(energy)]

    return torch.stack(output_set)
