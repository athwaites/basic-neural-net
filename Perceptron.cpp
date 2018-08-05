///////////////////////////////////////////////////////////////////////////////
//
// Perceptron.cpp
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

#include <cstdlib>
#include <limits>

#include "Perceptron.h"

c_Perceptron::c_Perceptron() :
	m_Inputs(NULL),
	m_Target(NULL),
	m_WeightedDeltaSum(NULL),
	m_SumProducts(0.0),
	m_Output(0.0),
	m_Delta(0.0),
	m_TrainRate(0.0),
	m_ActType(ACT_TANH)
{
}

c_Perceptron::c_Perceptron(const c_Perceptron &src)
{
	_copy(src);
}

c_Perceptron::~c_Perceptron()
{
}

void c_Perceptron::setWeights()
{
	// Randomly seed the weights
	for (size_t i = 0; i < m_Weights.size(); ++i) {
		m_Weights[i] = static_cast<double>(rand()) / RAND_MAX;
	}
}

void c_Perceptron::setWeights(const std::valarray<double> &weights)
{
	if ((m_Inputs != NULL) && (weights.size() != m_Inputs->size())) {
		// Weights do not match the size of the input array, so set the first N values of the weights array
		m_Weights[std::slice(0, weights.size(), 1)] = weights;
	} else {
		m_Weights = weights;
	}
}

void c_Perceptron::setInputs(const std::valarray<double> &inputs)
{
	m_Inputs = &inputs;
	if (m_Weights.size() != inputs.size()) {
		// Weights do not match the size of the input array, so resize accordingly
		m_Weights.resize(inputs.size());
		m_WeightedDeltas.resize(inputs.size());
	}
}

void c_Perceptron::setWeightedDeltaSum(const double &weightedDeltaSum)
{
	m_WeightedDeltaSum = &weightedDeltaSum;
	// Clear the Target
	m_Target = NULL;
}

void c_Perceptron::setTarget(const double &target)
{
	m_Target = &target;
	// Clear the Weighted Delta Sum
	m_WeightedDeltaSum = NULL;
}

void c_Perceptron::setTrainRate(const double &trainRate)
{
	m_TrainRate = trainRate;
}

void c_Perceptron::setActivation(const e_Activation &actType)
{
	m_ActType = actType;
}

double& c_Perceptron::operator[](const size_t &idx)
{
	return m_Weights[idx];
}

const size_t c_Perceptron::getSize()
{
	return m_Weights.size();
}

const double& c_Perceptron::getOutput()
{
	return m_Output;
}

const std::valarray<double>& c_Perceptron::getWeights()
{
	return m_Weights;
}

const std::valarray<double>& c_Perceptron::getWeightedDeltas()
{
	return m_WeightedDeltas;
}

bool c_Perceptron::evaluate()
{
	// Evaluate the perceptron response
	if (_calcSumProducts()) {
		_calcActivation();
		return true;
	}
	return false;
}

bool c_Perceptron::train()
{
	// Train the perceptron
	_calcDelta();
	_calcNewWeights();
	return true;
}

void c_Perceptron::_copy(const c_Perceptron &src)
{
	// Copy member variables
	m_Inputs = src.m_Inputs;
	m_Weights = src.m_Weights;
	m_WeightedDeltas = src.m_WeightedDeltas;
	m_Target = src.m_Target;
	m_WeightedDeltaSum = src.m_WeightedDeltaSum;
	m_SumProducts = src.m_SumProducts;
	m_Output = src.m_Output;
	m_Delta = src.m_Delta;
	m_TrainRate = src.m_TrainRate;
	m_ActType = src.m_ActType;
}

bool c_Perceptron::_calcSumProducts()
{
	if (m_Inputs != NULL) {
		m_SumProducts = (m_Weights * (*m_Inputs)).sum();
		return true;
	}
	return false;
}

bool c_Perceptron::_calcNewWeights()
{
	std::valarray<double> m_DeltaWeights;
	if (m_Inputs != NULL) {
		// Delta already determined; apply it to the weights to determine the weighted Deltas
		m_WeightedDeltas = m_Delta * m_Weights;
		m_Weights += m_TrainRate * m_Delta * (*m_Inputs);
		return true;
	}
	return false;
}

void c_Perceptron::_calcActivation()
{
	// Perform activation function based on activation type
	switch (m_ActType) {
	case ACT_TANH:
		// Activate using the tanh function [-1:0:1]
		m_Output = tanh((m_SumProducts / 2.0));
		break;
	case ACT_SIGMOID:
		// Activate using the sigmoid function [0:0.5:1]
		m_Output = (1.0 / (1.0 + exp(-m_SumProducts)));
		break;
	default:
		m_Output = m_SumProducts;
		break;
	}
}

double c_Perceptron::_calcActivDeriv()
{
	// Calculate activation derivative function based on activation type
	switch (m_ActType) {
	case ACT_TANH:
		// Calculate from derivative of the tanh function (sech^2(x))
		return pow((2.0 * cosh(m_SumProducts) / (cosh(2.0 * m_SumProducts) + 1)), 2);
	case ACT_SIGMOID:
		// Calculate from derivative of the sigmoid function (e^x / (1 + e^x)^2)
		return exp(m_SumProducts) / pow((exp(m_SumProducts) + 1.0), 2);
	default:
		return m_SumProducts;
	}
}

void c_Perceptron::_calcDelta()
{
	// First calculate the derivative of activation type
	m_Delta = _calcActivDeriv();
	// Second calculate delta from the activation derivative multiplied by the error
	// Check the type of delta (output layer perceptron, or backpropagation perceptron)
	if (m_Target != NULL) {
		// Target set, output layer perceptron
		m_Delta *= (*m_Target) - m_Output;
	} else if (m_WeightedDeltaSum != NULL) {
		// WeightedDeltaSum set, backpropagation perceptron
		m_Delta *= (*m_WeightedDeltaSum);
	} else {
		// No Target or Deltas set, cannot train!
		m_Delta = 0.0;
	}
}

bool c_Perceptron::_fcmp(const double &lhs, const double &rhs)
{
	return (fabs(lhs - rhs) < std::numeric_limits<double>::epsilon());
}



