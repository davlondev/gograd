package gograd

import (
	"fmt"
	"testing"
)

func TestBasicOperations(t *testing.T) {
	fmt.Println("\nTestBasicOperations")
	a := NewTensor(3)
	b := NewTensor(5)

	c := a.Add(b)
	fmt.Println(c)

	c = c.Mul(NewTensor(2))
	fmt.Println(c)

	c = c.Pow(NewTensor(2))
	fmt.Println(c)
}

func TestReLU(t *testing.T) {
	fmt.Println("\nTestReLU")
	a := NewTensor(3)
	a = a.ReLU()
	fmt.Println(a)

	a = NewTensor(-3)
	a = a.ReLU()
	fmt.Println(a)
}

func TestBack(t *testing.T) {
	fmt.Println("\nTestBack")

	a := NewTensor(3)
	b := NewTensor(4)

	a = a.Add(b)
	a = a.Mul(NewTensor(5))

	a = a.ReLU()
	a = a.Add(NewTensor(2))

	a.Backward()

	fmt.Println(a)
}
