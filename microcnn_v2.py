import math
import numpy as np
from graphviz import Digraph
import random
random.seed(42)


class Value:
  def __init__(self, data, _parents=(), _op='', label=''):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._prev = set(_parents)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
      #self._backward()
      #other._backward()

    out._backward = _backward
    return out

  def __radd__(self,other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
      #self._backward()
      #other._backward()
    out._backward = _backward
    return out

  def __pow__(self,other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += other*(self.data**(other-1))*out.grad
    out._backward = _backward

    return out

  def nat_log(self):
    out = Value(math.log(self.data), (self,), f'log({self.data})')

    def _backward():
      self.grad += (1/self.data)*out.grad
    out._backward = _backward

    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self,other):
    return self * other**-1

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/ (math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
      self._backward()


    out._backward = _backward
    return out

  def relu(self):
    x = self.data
    r = max(0,x)
    out = Value(r, (self, ), 'relu')

    def _backward():
      if r > 0:
        self.grad += 1 * out.grad
      else:
        self.grad += 0 * out.grad

      self._backward()

    out._backward = _backward
    return out

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def backward(self):

    #topological order of all the children in the graph

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1
    for v in reversed(topo):
      v._backward()

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other +(-self)

  def __int__(self):
    return self.data

  def __float__(self):
    return self.data

  def __gt__(self, other):
    # return self.data > Value(other)
    if(self.data > other):
      return True
    else:
      return False

  def __lt__(self, other):
    if (self.data < other):
      return True
    else:
      return False

  def __le__(self, other):
    if (self.data <= other):
      return True
    else:
      return False

  def __ge__(self, other):
    if (self.data >= other):
      return True
    else:
      return False


def trace(root):
  nodes, edges = set(), set()

  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child,v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format="svg", graph_attr={'rankdir': "LR"})

  nodes, edges = trace(root)

  for n in nodes:
    uid = str(id(n))

    dot.node(name=uid, label= "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

    if n._op:
      dot.node(name = uid + n._op, label = n._op)

      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot



class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    # out = act.tanh()
    out = act.relu()
    return out

  def parameter(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [value for neuron in self.neurons for value in neuron.parameter()]

class OutNeuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    # out = act.tanh()
    # out = act.relu()
    return act

  def parameter(self):
    return self.w + [self.b]

class OutLayer:
  def __init__(self, nin, nout):
    self.neurons = [OutNeuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [value for neuron in self.neurons for value in neuron.parameter()]

class MLP:

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [value for layer in self.layers for value in layer.parameters()]


class Filter:

  def __init__(self, k_size, depth, params=None):
    if (not params):
      self.w = np.array([Value(random.uniform(-1,1)) for _ in range(k_size*k_size*depth)]).reshape((depth, k_size, k_size))
      self.b = Value(random.uniform(-1,1))
    else:
      self.w = np.array(params[:-1]).reshape((depth, k_size, k_size))
      self.b = params[-1]

  def __call__(self, r_field):
    # w * x + b for a receptive field
    act = sum((r_field.flatten() * self.w.flatten()), self.b)
    # out = act.tanh()
    out = act.relu()
    return out

  def parameter(self):
    # returns a list containing all the weights + the bias
    return self.w.flatten().tolist() + [self.b]


class CLayer:

  def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, params = None):
    self.out_channels = out_channels
    self.stride = stride
    self.padding = padding
    self.k_size = k_size
    if (not params):
      self.filters = [Filter(k_size, in_channels) for _ in range(out_channels)]
    else:
      self.filters = []
      i = 0
      for _ in range(out_channels):
        f_params = params[i: i + k_size * k_size * in_channels + 1]
        i = i + k_size * k_size * in_channels + 1
        self.filters.append(Filter(k_size, in_channels, f_params))

  def __call__(self,x):
    out_w = ((np.shape(x)[1] + 2*self.padding - self.k_size)//self.stride) + 1
    x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))
    out = []
    for filt in self.filters:
        for i in range(out_w):
           for j in range(out_w):
               tl = self.stride * i, self.stride * j
               out.append(filt(x[:, tl[0]:tl[0] + self.k_size, tl[1]:tl[1] + self.k_size]))

    return np.array(out).reshape((self.out_channels, out_w, out_w))

  def parameters(self):
    return [value for filt in self.filters for value in filt.parameter()]


class MaxPool:

  def __init__(self, k_size, stride):
    self.k_size = k_size
    self.stride = stride

  def __call__(self, x):
    out_w = ((np.shape(x)[1] - self.k_size)//self.stride) + 1
    out_channels = np.shape(x)[0]
    out = []
    for channel in x:
      for i in range(out_w):
        for j in range(out_w):
          tl = self.stride * i, self.stride * j
          out.append(max(channel[tl[0]:tl[0] + self.k_size, tl[1]:tl[1] + self.k_size].flatten()))

    return np.array(out).reshape(out_channels, out_w, out_w)
