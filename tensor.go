package gograd

/*
	Based on Karpathy's Micrograd
	Credit:  https://github.com/karpathy/micrograd/
*/

import (
	"fmt"
	"math"
)

type Tensor struct {
	data  float32
	grad  float32
	op    string
	_back func()
	_prev []Tensor
}

func (t Tensor) String() string {
	s := fmt.Sprintf("Tensor{data=%f, grad=%f, op=%s}", t.data, t.grad, t.op)
	return s
}

func NewTensor(data float32) Tensor {
	return Tensor{
		data:  data,
		grad:  0.0,
		op:    "",
		_back: nil,
		_prev: nil,
	}
}

func (t Tensor) Add(o Tensor) Tensor {
	out := Tensor{
		data:  t.data + o.data,
		grad:  0.0,
		op:    "add",
		_back: nil,
		_prev: []Tensor{t, o},
	}

	out._back = func() {
		t.grad += out.grad
		o.grad += out.grad
	}

	return out
}

func (t Tensor) Mul(o Tensor) Tensor {
	out := Tensor{
		data:  t.data * o.data,
		grad:  0.0,
		op:    "mul",
		_back: nil,
		_prev: []Tensor{t, o},
	}

	out._back = func() {
		t.grad += o.grad * out.grad
		o.grad += t.grad * out.grad
	}

	return out
}

func (t Tensor) Pow(o Tensor) Tensor {
	out := Tensor{
		data:  float32(math.Pow(float64(t.data), float64(o.data))),
		grad:  0.0,
		op:    "pow",
		_back: nil,
		_prev: []Tensor{t},
	}
	return out
}

func (t Tensor) ReLU() Tensor {
	out := NewTensor(0)
	if t.data < 0.0 {
		out = Tensor{
			data:  0.0,
			grad:  0.0,
			op:    "ReLU",
			_back: nil,
			_prev: nil,
		}
	} else {
		out = Tensor{
			data:  t.data,
			grad:  0.0,
			op:    "ReLU",
			_back: nil,
			_prev: nil,
		}
	}

	out._back = func() {
		if out.grad > 0 {
			t.grad += out.grad
		}
	}

	return out
}

func (t *Tensor) Backward() {
	topo := []Tensor{}
	visited := []Tensor{}

	build_topo := func(tt Tensor) {}
	build_topo = func(tt Tensor) {
		if !MatchFound(visited, &tt) {
			visited = append(visited, tt)
			for _, child := range tt._prev {
				build_topo(child)
			}
			topo = append(topo, tt)
		}
	}
	build_topo(*t)

	t.grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i]._back != nil {
			topo[i]._back()
		}
	}
}

func MatchFound(arr []Tensor, t *Tensor) bool {
	for _, i := range arr {
		if &i == t {
			return true
		}
	}
	return false
}
