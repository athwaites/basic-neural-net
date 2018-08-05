///////////////////////////////////////////////////////////////////////////////
//
// PerceptronLayer.cpp
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

#include "PerceptronLayer.h"

c_PerceptronLayer::c_PerceptronLayer(const size_t &numPerceptrons) :
	m_Inputs(NULL),
	m_Targets(NULL),
	m_WeightedDeltaSumsIn(NULL),
	m_Input(NULL),
	m_Output(NULL),
	m_Bias(true),
	m_Size(numPerceptrons),
	m_TrainRate(0.0),
	m_ActType(ACT_TANH)
{
	_build();
}

c_PerceptronLayer::c_PerceptronLayer(const c_PerceptronLayer &src)
{
	_copy(src);
}

c_PerceptronLayer::~c_PerceptronLayer()
{
}

void c_PerceptronLayer::setInput(c_PerceptronLayer &input)
{
	m_Input = &input;
	_build();
}

void c_PerceptronLayer::setInputs(const std::valarray<double> &inputs)
{
	m_Inputs = &inputs;
	_connectInputs();
}

void c_PerceptronLayer::setTargets(const std::valarray<double> &targets)
{
	if (targets.size() == m_Size) {
		m_Targets = &targets;
		// Clear the Weighted Delta Sums
		m_WeightedDeltaSumsIn = NULL;
		_connectTargets();
	}
}

void c_PerceptronLayer::setTrainRate(const double &trainRate)
{
	m_TrainRate = trainRate;
	// Apply the new training rate to all perceptrons in the layer
	for (size_t i = 0; i < m_Size; ++i) {
		m_Perceptrons[i].setTrainRate(trainRate);
	}
}

void c_PerceptronLayer::setActivation(const e_Activation &actType)
{
	m_ActType = actType;
	// Apply the new activation type to all perceptrons in the layer
	for (size_t i = 0; i < m_Size; ++i) {
		m_Perceptrons[i].setActivation(actType);
	}
}

void c_PerceptronLayer::setBias(const bool &bias)
{
	m_Bias = bias;
	_build();
}

c_Perceptron& c_PerceptronLayer::operator[](const size_t &idx)
{
	return m_Perceptrons[idx];
}

const size_t c_PerceptronLayer::getSize()
{
	return m_Size;
}

const std::valarray<double>& c_PerceptronLayer::getOutputs()
{
	return m_Outputs;
}

const std::valarray<double>& c_PerceptronLayer::getWeightedDeltaSumsOut()
{
	return m_WeightedDeltaSumsOut;
}

void c_PerceptronLayer::evaluate()
{
	// Evaluate the perceptron layer
	for (size_t i = 0; i < m_Size; ++i) {
		if (m_Perceptrons[i].evaluate()) {
			m_Outputs[i] = m_Perceptrons[i].getOutput();
		}
	}
}

void c_PerceptronLayer::train()
{
	// Train the perceptron layer
	m_WeightedDeltaSumsOut = 0;
	for (size_t i = 0; i < m_Size; ++i) {
		if (m_Perceptrons[i].train()) {
			m_WeightedDeltaSumsOut += m_Perceptrons[i].getWeightedDeltas();
		}
	}
}

void c_PerceptronLayer::_setOutput(c_PerceptronLayer &output)
{
	m_Output = &output;
}

void c_PerceptronLayer::_setWeightedDeltaSumsIn(const std::valarray<double> &weightedDeltaSums)
{
	m_WeightedDeltaSumsIn = &weightedDeltaSums;
	// Clear the Targets
	m_Targets = NULL;
	// Connect these weighted delta sums to the perceptrons
	_connectWeightedDeltaSums();
}

void c_PerceptronLayer::_copy(const c_PerceptronLayer &src)
{
	// Copy member variables
	m_Inputs = src.m_Inputs;
	m_Targets = src.m_Targets;
	m_WeightedDeltaSumsIn = src.m_WeightedDeltaSumsIn;
	m_Input = src.m_Input;
	m_Output = src.m_Output;
	m_Outputs = src.m_Outputs;
	m_WeightedDeltaSumsOut = src.m_WeightedDeltaSumsOut;
	m_Perceptrons = src.m_Perceptrons;
	m_Bias = src.m_Bias;
	m_Size = src.m_Size;
	m_TrainRate = src.m_TrainRate;
	m_ActType = src.m_ActType;
	// Connect the new perceptrons correctly
	_connect();
}

void c_PerceptronLayer::_build()
{
	// Resize this layer to the number of perceptrons specified
	m_Perceptrons.resize(m_Size);
	// Resize the output array
	if (m_Bias) {
		// Bias is enabled, so increase the output array by one extra element (for bias node)
		m_Outputs.resize(m_Size + 1);
		// Set the bias node to 1
		m_Outputs[m_Size] = 1.0;
	} else {
		m_Outputs.resize(m_Size);
	}
	if (m_Input != NULL) {
		// Set the inputs array to the input layer's output array
		setInputs(m_Input->getOutputs());
		// Set the input layer's output layer to this
		m_Input->_setOutput(*this);
		// Resize the weighted delta sums to match the input layer's size
		m_WeightedDeltaSumsOut.resize(m_Input->getSize());
		// Set the input layer's weighted delta sums in to this layer's weighted delta sums out
		m_Input->_setWeightedDeltaSumsIn(m_WeightedDeltaSumsOut);
	}
	if (m_Output != NULL) {
		// Resize the weighted delta sums of the output layer
		m_Output->m_WeightedDeltaSumsOut.resize(m_Size);
	}
	_connect();
}

void c_PerceptronLayer::_connect()
{
	// Connect the perceptrons correctly
	_connectInputs();
	_connectTargets();
	_connectWeightedDeltaSums();
}

void c_PerceptronLayer::_connectInputs()
{
	if (m_Inputs != NULL) {
		// Check the perceptron inputs are available then set for each perceptron
		for (size_t i = 0; i < m_Size; ++i) {
			m_Perceptrons[i].setInputs((*m_Inputs));
		}
	}
}

void c_PerceptronLayer::_connectTargets()
{
	if ((m_Targets != NULL) && (m_Targets->size() == m_Size)) {
		// Check the perceptron targets are available and the right size then set for each perceptron
		for (size_t i = 0; i < m_Size; ++i) {
			m_Perceptrons[i].setTarget((*m_Targets)[i]);
		}
	}
}

void c_PerceptronLayer::_connectWeightedDeltaSums()
{
	if ((m_WeightedDeltaSumsIn != NULL) && (m_WeightedDeltaSumsIn->size() == m_Size)) {
		// Check the weighted delta sums are available and the right size then set for each perceptron
		for (size_t i = 0; i < m_Size; ++i) {
			m_Perceptrons[i].setWeightedDeltaSum((*m_WeightedDeltaSumsIn)[i]);
		}
	}
}
