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

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include "neural_net_constants.h"
#include "neural_net.h"

float TanhScaled(float x){
  return 1.7159*tanh(0.66666667*x);
}

float TanhScaledPrime(float x) {
  return 0.66666667/1.7159*(1.7159+tanh(x))*(1.7159-tanh(x));
}

float Sigmoid(float x) {
  return 1/(1+pow(exp(1.0), -x));
}

float SigmoidPrime(float x) {
  return Sigmoid(x) * (1.0-Sigmoid(x));
}

int main () {
  float training_data_arr[] = {
    1.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f
  };
  int batch_size = 4;
  int n_input = 2;
  int n_hidden = 8;
  int n_output = 1;
  int n_iterations = 500;
  int n_converged = 0;
  int total_epoch = 0;
  long long elapsed_time  = 0;
  for (int iteration = 0; iteration < n_iterations; iteration++) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    neuralplex::NeuralNet *neural_net = new neuralplex::NeuralNet(n_input, n_hidden, n_output, Sigmoid, SigmoidPrime);
    float global_error = Train(training_data_arr, batch_size, neuralplex::kLearningAlgorithmsResilientProp);
    if (global_error <= neuralplex::kNeuralLearningThreshold) {
      n_converged++;
      total_epoch += neural_net->epoch();
      gettimeofday(&end, NULL);
      elapsed_time +=   (end.tv_sec * (unsigned int)1e6 +   end.tv_usec) - (start.tv_sec * (unsigned int)1e6 + start.tv_usec);
    }
  }
  std::cout << ( (float)n_converged/(float)n_iterations*100.0f) << "%" << " network convergence. " << std::endl;
  if (n_converged > 0) {
    std::cout << (total_epoch/n_converged) << " avg training intervals (epoch) to convergence." << std::endl;
    std::cout << (elapsed_time/n_converged) << "ms avg time to converge." << std::endl;
  }
  return 0;
}

