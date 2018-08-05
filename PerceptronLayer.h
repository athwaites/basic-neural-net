///////////////////////////////////////////////////////////////////////////////
//
// PerceptronLayer.h
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

#ifndef PERCEPTRONLAYER_H_
#define PERCEPTRONLAYER_H_

#include <valarray>
#include <vector>

#include "Perceptron.h"

class c_PerceptronLayer {
public:
	// Constructors
									c_PerceptronLayer(const size_t &numPerceptrons);
									c_PerceptronLayer(const c_PerceptronLayer &src);
	// Destructor
	virtual							~c_PerceptronLayer();
	// Set
	void							setInput(c_PerceptronLayer &input);
	void							setInputs(const std::valarray<double> &inputs);
	void							setTargets(const std::valarray<double> &targets);
	void							setTrainRate(const double &trainRate);
	void							setActivation(const e_Activation &actType);
	void							setBias(const bool &bias);
	// Get
	c_Perceptron&					operator[](const size_t &idx);
	const size_t					getSize();
	const std::valarray<double>&	getOutputs();
	const std::valarray<double>&	getWeightedDeltaSumsOut();
	// Functions
	void							evaluate();
	void							train();
private:
	// Functions
	void							_setOutput(c_PerceptronLayer &output);
	void							_setWeightedDeltaSumsIn(const std::valarray<double> &weightedDeltaSums);
	void							_copy(const c_PerceptronLayer &src);
	void							_build();
	void							_connect();
	void							_connectInputs();
	void							_connectTargets();
	void							_connectWeightedDeltaSums();
	// Variables
	const std::valarray<double>		*m_Inputs;
	const std::valarray<double>		*m_Targets;
	const std::valarray<double>		*m_WeightedDeltaSumsIn;
	c_PerceptronLayer				*m_Input;
	c_PerceptronLayer				*m_Output;
	std::valarray<double>			m_Outputs;
	std::valarray<double>			m_WeightedDeltaSumsOut;
	std::vector<c_Perceptron>		m_Perceptrons;
	bool							m_Bias;
	size_t							m_Size;
	double							m_TrainRate;
	e_Activation					m_ActType;
};

#endif PERCEPTRONLAYER_H_
