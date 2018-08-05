///////////////////////////////////////////////////////////////////////////////
//
// NeuralNetwork.cpp
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

#include "NeuralNetwork.h"

c_NeuralNetwork::c_NeuralNetwork(const std::valarray<double> &inputs, const std::valarray<double> &targets, const std::vector<size_t> &layers, const e_Activation &actType, const double &trainRate, const bool &bias) :
	m_Inputs(&inputs),
	m_Targets(&targets),
	m_Size(layers.size())
{
	_build(layers);
	setBias(bias);
	setTrainRate(trainRate);
	setActivation(actType);
}

c_NeuralNetwork::c_NeuralNetwork(const c_NeuralNetwork &src)
{
	_copy(src);
}

c_NeuralNetwork::~c_NeuralNetwork()
{
}

void c_NeuralNetwork::setInputs(const std::valarray<double> &inputs)
{
	m_Inputs = &inputs;
	_resizeLocalInputs();
	_connectInputs();
}

void c_NeuralNetwork::setTargets(const std::valarray<double> &targets)
{
	m_Targets = &targets;
	_connectTargets();
}

void c_NeuralNetwork::setTrainRate(const double &trainRate)
{
	m_TrainRate = trainRate;
	// Apply the new training rate to all layers in the network
	for (size_t i = 0; i < m_Size; ++i) {
		m_Layers[i].setTrainRate(trainRate);
	}
}

void c_NeuralNetwork::setActivation(const e_Activation &actType)
{
	m_ActType = actType;
	// Apply the new activation type to all layers in the network
	for (size_t i = 0; i < m_Size; ++i) {
		m_Layers[i].setActivation(actType);
	}
}

void c_NeuralNetwork::setBias(const bool &bias)
{
	m_Bias = bias;
	// Apply the new bias state to all but the last layer in the network
	for (size_t i = 0; i < (m_Size - 1); ++i) {
		m_Layers[i].setBias(bias);
	}
	m_Layers[m_Size - 1].setBias(false);
	_resizeLocalInputs();
	_connectInputs();
}

c_PerceptronLayer& c_NeuralNetwork::operator[](const size_t &idx)
{
	return m_Layers[idx];
}

const size_t c_NeuralNetwork::getSize()
{
	return m_Size;
}

const std::valarray<double>& c_NeuralNetwork::getOutputs()
{
	return m_Layers[m_Size - 1].getOutputs();
}

void c_NeuralNetwork::evaluate()
{
	// Evaluate the network (feedforwards)
	if (m_Inputs != NULL) {
		_updateLocalInputs();
		for (size_t i = 0; i < m_Size; ++i) {
			m_Layers[i].evaluate();
		}
	}
}

void c_NeuralNetwork::train()
{
	// Train the network (backpropagation)
	if ((m_Inputs != NULL) && (m_Targets != NULL)) {
		_updateLocalInputs();
		for (size_t i = m_Size; i > 0; --i) {
			m_Layers[(i - 1)].train();
		}
	}
}

void c_NeuralNetwork::_updateLocalInputs()
{
	if (m_Bias) {
		// Set the first N values of the local inputs array (leaving the last one untouched)
		m_LocalInputs[std::slice(0, m_Inputs->size(), 1)] = (*m_Inputs);
		// Set the last local input to the bias value
		m_LocalInputs[m_LocalInputs.size() - 1] = 1.0;
	} else {
		// Copy the input array directly to the local inputs array
		m_LocalInputs = (*m_Inputs);
	}
}

void c_NeuralNetwork::_resizeLocalInputs()
{
	if (m_Inputs != NULL) {
		if (m_Bias) {
			m_LocalInputs.resize(m_Inputs->size() + 1);
		} else {
			m_LocalInputs.resize(m_Inputs->size());
		}
	}
}

void c_NeuralNetwork::_copy(const c_NeuralNetwork &src)
{
	// Copy member variables
	m_Inputs = src.m_Inputs;
	m_Targets = src.m_Targets;
	m_Layers = src.m_Layers;
	m_Bias = src.m_Bias;
	m_Size = src.m_Size;
	m_TrainRate = src.m_TrainRate;
	m_ActType = src.m_ActType;
	// Connect the new layers correctly
	_connect();
}

void c_NeuralNetwork::_build(const std::vector<size_t> &layers)
{
	// Build a single-layer or multi-layer neural network
	// Reserve the layers vector size
	m_Layers.reserve(m_Size);
	for (size_t i = 0; i < m_Size; ++i) {
		// For each required layer, push back a blank layer of the required size
		m_Layers.push_back(layers[i]);
	}
	// Connect the layers
	_connect();
}

void c_NeuralNetwork::_connect()
{
	// Connect the layers correctly
	_connectInputs();
	_connectTargets();
	_connectLayers();
}

void c_NeuralNetwork::_connectInputs()
{
	if (m_Layers.size() > 0) {
		// Connect the inputs to the first layer
		if (m_Inputs != NULL) {
			m_Layers[0].setInputs(m_LocalInputs);
		}
	}
}

void c_NeuralNetwork::_connectTargets()
{
	if (m_Layers.size() > 0) {
		// Connect the targets to the last layer
		if (m_Targets != NULL) {
			m_Layers[m_Size - 1].setTargets(*m_Targets);
		}
	}
}

void c_NeuralNetwork::_connectLayers()
{
	if (m_Layers.size() > 1) {
		// Connect the layers for all layers after the first
		for (size_t i = 1; i < m_Size; ++i) {
			m_Layers[i].setInput(m_Layers[i - 1]);
		}
	}
}



