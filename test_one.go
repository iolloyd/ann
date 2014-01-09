package main

import (
	"./nn"
)

var (
	inSize     = 4
	hiddenSize = 4
	outSize    = 2
	iterations = 100
	errorThreshold = 0.1
)

func main() {
	td := nn.TrainingData{}
	td.Inputs = [][]float64{{9.0, 1.0, 0.0, 1.0}}
	td.Outputs = [][]float64{{0.0, 1.0}}

	nn := nn.MakeNN(inSize, hiddenSize, outSize, errorThreshold)
	nn.Train(td, iterations)
	nn.Show()
}
