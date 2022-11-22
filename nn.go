package gograd

/*
	Based on Karpathy's Micrograd
	Credit:  https://github.com/karpathy/micrograd/
*/

import (
	"math/rand"
)

type Module struct {
	params []Tensor
}

func (m *Module) zero_grad() {
	for _, p := range m.params {
		p.grad = 0.0
	}
}

func (m Module) paramaters() []Tensor {
	return m.params
}

type Neuron struct {
	w         Tensor
	b         Tensor
	nonLinear bool
}

func NewNeuron(nin int, isLinear bool) Neuron {
	return Neuron{
		w:         NewTensor(rand.Float32()),
		b:         NewTensor(0),
		nonLinear: !isLinear,
	}
}

func Forward(x Tensor) float32 {
	act := float32(0.0)
	// todo
	return act
}
