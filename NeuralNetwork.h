///////////////////////////////////////////////////////////////////////////////
//
// NeuralNetwork.h
//
// Copyright (c) 2018 Adam Thwaites
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <valarray>
#include <vector>

#include "PerceptronLayer.h"

class c_NeuralNetwork {
public:
	// Constructors
									c_NeuralNetwork(const std::valarray<double> &inputs, const std::valarray<double> &targets, const std::vector<size_t> &layers, const e_Activation &actType, const double &trainRate, const bool &bias);
									c_NeuralNetwork(const c_NeuralNetwork &src);
	// Destructor
	virtual							~c_NeuralNetwork();
	// Set
	void							setInputs(const std::valarray<double> &inputs);
	void							setTargets(const std::valarray<double> &targets);
	void							setTrainRate(const double &trainRate);
	void							setActivation(const e_Activation &actType);
	void							setBias(const bool &bias);
	// Get
	c_PerceptronLayer&				operator[](const size_t &idx);
	const size_t					getSize();
	const std::valarray<double>&	getOutputs();
	// Functions
	void							evaluate();
	void							train();
private:
	// Functions
	void							_updateLocalInputs();
	void							_resizeLocalInputs();
	void							_copy(const c_NeuralNetwork &src);
	void							_build(const std::vector<size_t> &layers);
	void							_connect();
	void							_connectInputs();
	void							_connectTargets();
	void							_connectLayers();
	// Variables
	const std::valarray<double>		*m_Inputs;
	const std::valarray<double>		*m_Targets;
	std::valarray<double>			m_LocalInputs;
	std::vector<c_PerceptronLayer>	m_Layers;
	bool							m_Bias;
	size_t							m_Size;
	double							m_TrainRate;
	e_Activation					m_ActType;
};

#endif NEURALNETWORK_H_

