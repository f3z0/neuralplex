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
#include <time.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include <unistd.h>
#include <exception>
#include <random>
#include <climits>
#include <cfloat>
#include "neural_net.h"
#include "neural_net_constants.h"
#include "rapidjson/filestream.h"

namespace neuralplex {

NeuralNet::NeuralNet (int n_input, int n_hidden, int n_output, float (*activation)(float), float (*activation_p)(float)) {
  try {
    n_input_ = n_input;
    n_hidden_ = n_hidden;
    n_output_ = n_output;
    static std::random_device rd;
    static std::mt19937_64 mt(rd());
    static std::uniform_real_distribution<float> distribution(-1.0/sqrt(n_input), 1.0/sqrt(n_input));
    int n_weights = n_output + (n_output * n_hidden) + (n_input * n_hidden) + n_hidden;
    float start_weights[n_weights];
    for(int x = 0; x < n_weights; x++) start_weights[x] = distribution(mt);
    BuildNetwork(activation, activation_p, &start_weights[0]);
    max_float_training_ = 0.0f;
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

NeuralNet::NeuralNet (int n_input, int n_hidden, int n_output, float (*activation)(float), float (*activation_p)(float), float *start_weights) {
  try {
    n_input_ = n_input;
    n_hidden_ = n_hidden;
    n_output_ = n_output;
    BuildNetwork(activation, activation_p, start_weights);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

NeuralNet::~NeuralNet() { }

float NeuralNet::Train(float training_data[], int batch_size, int learning_algo) {
  float mse = 1.0f;
  epoch_ = 0;
  max_float_training_ = FLT_MIN;
  min_float_training_ = FLT_MAX;
  for (int row = 0; row < batch_size * (n_input_+n_output_); row += (n_input_ + n_output_)) {
    for (int x = 0; x < n_input_; x++) {
      if (training_data[row+x] > max_float_training_) {
//        std::cout << training_data[row+x] << std::endl;
          max_float_training_ = training_data[row+x];
}
      if (training_data[row+x] < min_float_training_) min_float_training_ = training_data[row+x];
    }
  }
  //std::cout << "MAX1 " << min_float_training_ << std::endl;

  NormalizeInputs(&training_data[0], batch_size);
  while (mse > kNeuralLearningThreshold && epoch_ < kNeuralLearningMaxEpoch) {
    mse = 0.0f;
    for(int row = 0; row < batch_size*(n_input_+n_output_); row += (n_input_ + n_output_)) {
      sort( neurons_.begin(), neurons_.end(), ForwardPropagation() );
      for(int x = 0; x < n_input_; x++) input_neurons_[x]->set_input(training_data[row+x]);
      for(int x = 0; x < n_output_; x++) output_neurons_[x]->set_ideal(training_data[row+n_input_+x]);
      for(std::vector<Neuron*>::iterator it = neurons_.begin(); it != neurons_.end(); ++it) (*it)->Forward();
      sort( neurons_.begin(), neurons_.end(), BackPropagation() );
      for(std::vector<Neuron*>::iterator it = neurons_.begin(); it != neurons_.end(); ++it) (*it)->Backward();
      for(std::vector<Neuron*>::iterator it = output_neurons_.begin(); it != output_neurons_.end(); ++it) mse += pow((*it)->error(),2)/n_output_;
    }
    for(std::vector<Neuron*>::iterator it = neurons_.begin(); it != neurons_.end(); ++it) (*it)->Learn(learning_algo);
    mse /= batch_size;
    std::cout << epoch_ << " " << "MSE: " << mse << std::endl;
    epoch_++;
  }
  return mse;
}

void NeuralNet::Compute(float inputs[], float* outputs) {
  try {
    NormalizeInputs(inputs, 1);
    sort( neurons_.begin(), neurons_.end(), ForwardPropagation() );
    for(int x = 0; x < n_input_; x++) {input_neurons_[x]->set_input(inputs[x]);}
    for(std::vector<Neuron*>::iterator it = neurons_.begin(); it != neurons_.end(); ++it) (*it)->Forward();
    for(int x = 0; x < n_output_; x++) outputs[x] = output_neurons_[x]->output();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

void NeuralNet::BuildNetwork(float (*activation)(float), float (*activation_p)(float), float *start_weights){
  try {
    Neuron *bias_neuron = new Neuron::Neuron("b0", activation, activation_p);
    neurons_.push_back(bias_neuron);
    bias_neuron->set_input(1.0f);
    bias_neurons_.push_back(bias_neuron);
    for (int i=0; i < n_output_; i++) {
      Neuron *output_neuron = new Neuron::Neuron("o"+std::to_string(i), activation, activation_p);
      neurons_.push_back(output_neuron);
      bias_neuron->ConnectTo(output_neuron, *start_weights++); 
      output_neurons_.push_back(output_neuron);
    }
    bias_neuron = new Neuron::Neuron("b1", activation, activation_p);
    neurons_.push_back(bias_neuron);
    bias_neurons_.push_back(bias_neuron);
    bias_neuron->set_input(1.0f);
    for (int i=0; i < n_hidden_;i++) {
      Neuron *hidden_neuron = new Neuron::Neuron("h"+std::to_string(i), activation, activation_p);
      neurons_.push_back(hidden_neuron);
      bias_neuron->ConnectTo(hidden_neuron, *start_weights++); 
      hidden_neurons_.push_back(hidden_neuron);
      for (int x = 0; x < output_neurons_.size(); x++) hidden_neuron->ConnectTo(output_neurons_[x], *start_weights++); 
    }
    for (int i=0; i < n_input_; i++){
      Neuron *input_neuron = new Neuron::Neuron("i"+std::to_string(i), activation, activation_p);
      neurons_.push_back(input_neuron);
      for(int x = 0; x < hidden_neurons_.size(); x++) input_neuron->ConnectTo(hidden_neurons_[x], *start_weights++);  
      input_neurons_.push_back(input_neuron);
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

void NeuralNet::NormalizeInputs(float* training_data, int batch_size) {
 // std::cout << max_float_training_ << std::endl;
 //std::cout << "MAX2 " << min_float_training_ << std::endl;

  for (int row = 0; row < batch_size*(n_input_+n_output_); row += (n_input_+n_output_)) {
    for (int x = 0; x < n_input_; x++) training_data[row+x] = (float)training_data[row+x] * ( kNeuralInputRange/(max_float_training_-min_float_training_)  ) + 
      (kNeuralInputLower - (min_float_training_*(kNeuralInputRange / (max_float_training_-min_float_training_))));
  }
}
} //namespace neuralplex