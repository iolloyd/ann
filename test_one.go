package main

import (
	"fmt"
	"./nn"
)

var (
	inSize     = 4
	hiddenSize = 4
	outSize    = 2
	iterations = 8 
	errorThreshold = 0.001
	rate = 0.5
)

func main() {
	td := nn.TrainingData{}
	td.Inputs = [][]float64{{9.0, 1.0, 0.0, 1.0}}
	td.Outputs = [][]float64{{0.0, 1.0}}

	fmt.Println("Targets are ", td.Outputs)
	nn := nn.MakeNN(inSize, hiddenSize, outSize, errorThreshold)
	nn.Train(td, iterations, rate)
	nn.Show()
}
