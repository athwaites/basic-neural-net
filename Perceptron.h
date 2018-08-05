///////////////////////////////////////////////////////////////////////////////
//
// Perceptron.h
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

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <valarray>

enum e_Activation {
	ACT_TANH,
	ACT_SIGMOID
};

class c_Perceptron {
public:
	// Constructors
									c_Perceptron();
									c_Perceptron(const c_Perceptron &src);
	// Destructor
	virtual							~c_Perceptron();
	// Set
	void							setWeights();
	void							setWeights(const std::valarray<double> &weights);
	void							setInputs(const std::valarray<double> &inputs);
	void							setWeightedDeltaSum(const double &weightedDeltaSum);
	void							setTarget(const double &target);
	void							setTrainRate(const double &trainRate);
	void							setActivation(const e_Activation &actType);
	// Get
	double&							operator[](const size_t &idx);
	const size_t					getSize();
	const double&					getOutput();
	const double&					getDelta();
	const std::valarray<double>&	getWeights();
	const std::valarray<double>&	getWeightedDeltas();
	// Functions
	bool							evaluate();
	bool							train();
private:
	// Functions
	void							_copy(const c_Perceptron &src);
	bool							_calcSumProducts();
	bool							_calcNewWeights();
	void							_calcActivation();
	double							_calcActivDeriv();
	void							_calcDelta();
	bool							_fcmp(const double &lhs, const double &rhs);
	// Variables
	const std::valarray<double>		*m_Inputs;
	std::valarray<double>			m_Weights;
	std::valarray<double>			m_WeightedDeltas;
	const double					*m_Target;
	const double					*m_WeightedDeltaSum;
	double							m_SumProducts;
	double							m_Output;
	double							m_Delta;
	double							m_TrainRate;
	e_Activation					m_ActType;
};

#endif PERCEPTRON_H_

