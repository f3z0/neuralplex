// neuralplex is distributed under BSD license reproduced below.
//
// Copyright (c) 2015 Gregory "f3z0" Ray, f3z0@fezo.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the psutil authors nor the names of its contributors
//    may be used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef NEURAL_NET_CONSTANTS_H_
#define NEURAL_NET_CONSTANTS_H_

namespace neuralplex {

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

enum LearningAlgorithms {
  kLearningAlgorithmsUndefined = 0,
  kLearningAlgorithmsBackProp,
  kLearningAlgorithmsResilientProp
};

//const unsigned int kMaxBatchSize = 20;
const float kNeuralInputUpper = 1.0f;
const float kNeuralInputLower = -1.0f;
const float kNeuralInputRange = kNeuralInputUpper - kNeuralInputLower;
const int kNeuralLearningMaxEpoch = 1000000;
const float kBackPropLearningRate = 0.7;
const float kBackPropMomentum = 0.7;
const float kNeuralLearningThreshold = 0.03;
const float kResilientPropInitUpdateVal = 0.1;
const float kResilientPropDeltaMax = 50.0;
const float kResilientPropUpdateMin = 1e-6;
const float kResilientPropSlower = 0.5;
const float kResilientPropFaster = 1.2;

} //namespace neuralplex
#endif /*NEURAL_NET_CONSTANTS_H_*/
