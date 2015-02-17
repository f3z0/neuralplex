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
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <cmath>
#include <numeric>
#include "neural_net_constants.h"
#include "neural_net_exceptions.h"
#include "neuron.h"
#include "neural_net.h"

namespace neuralplex {

Neuron::Neuron(std::string name, float (*activation)(float), float (*activation_prime)(float)) {
  name_ = name;
  activation_ = activation;
  activation_prime_ = activation_prime;
  summation_ = 0.0f;
  layer_idx_ = 0;
  has_input_ = false;
  has_ideal_ = false;
  input_ = 0.0f;
  ideal_ = 0.0f;
  delta_ = 0.0f;
  error_ = 0.0f;
  output_ = 0.0f;
}

Neuron::~Neuron() {}

void Neuron::ConnectTo(Neuron *n, float weight) {
  n->set_layer_idx(layer_idx_+1);
  synapse_t child_synapse;
  child_synapse.parent = this;
  child_synapse.child = n;
  child_synapse.weight = weight;
  child_synapse.weight_delta = 0.0f;
  child_synapse.last_weight_delta = 0.0f;
  child_synapse.next_weight = child_synapse.weight;
  child_synapse.last_delta = 0.0f;
  child_synapse.last_update_val = kResilientPropInitUpdateVal;
  child_synapse.update_val = kResilientPropInitUpdateVal;
  children_.push_back(child_synapse);
  n->parents().push_back(child_synapse);
}

void Neuron::Forward() {
  if (has_input_){
    summation_ = input_;
    output_ = input_;
  } else {
    summation_ = 0.0;
    for (std::vector<synapse_t>::iterator it = parents_.begin(); it != parents_.end(); ++it) {
      summation_ += (*it).parent->output() * (*it).weight;
    }
    output_ = activation_(summation_);
  }
}

void Neuron::Backward() {
  delta_ = 0.0;
  if (has_ideal_) {
    error_ =   ideal_ - output_;
    delta_ = error_ * activation_prime_(output_);
  } else {
    for(std::vector<synapse_t>::iterator it = children_.begin(); it != children_.end(); ++it) delta_ += (*it).weight * (*it).child->delta();
    delta_ *= activation_prime_(summation_);
    for (std::vector<synapse_t>::iterator it = children_.begin(); it != children_.end(); ++it) {
      (*it).batch_gradients.push_back(output_ * (*it).child->delta());
      for (std::vector<synapse_t>::iterator it2 = (*it).child->parents().begin(); it2 != (*it).child->parents().end(); ++it2) {
        if ((*it2).parent == this && (*it2).child == (*it).child) (*it2).batch_gradients = (*it).batch_gradients;
      }
    }
  }
}

void Neuron::Learn(int learning_algo){
  switch (learning_algo) {
    case kLearningAlgorithmsBackProp:
      LearnBackProp();
      break;
    case kLearningAlgorithmsResilientProp:
      LearnRProp();
      break;
    default:
      throw UndefinedLearningAlgoException();
  } 
}

void Neuron::LearnBackProp() {
  if(!has_ideal_){
    for (std::vector<synapse_t>::iterator it = children_.begin(); it != children_.end(); ++it) {
      float gradient_batch_sum = 0.0f;
      for (int x = 0; x < (*it).batch_gradients.size(); x++) gradient_batch_sum += (*it).batch_gradients[x];
      (*it).batch_gradients.clear();
      (*it).last_delta = ((kBackPropLearningRate * gradient_batch_sum) + (kBackPropMomentum * (*it).last_delta));
      (*it).weight += (*it).last_delta;
      for (std::vector<synapse_t>::iterator it2 = (*it).child->parents().begin(); it2 != (*it).child->parents().end(); ++it2) {
        if((*it2).parent == this && (*it2).child == (*it).child) (*it2).weight = (*it).weight;
      }
    }
  }
}

void Neuron::LearnRProp() {
  if(!has_ideal_){
    for (std::vector<synapse_t>::iterator it = children_.begin(); it != children_.end(); ++it) {
      float gradient_batch_sum = 0.0f;
      for (int x = 0; x < (*it).batch_gradients.size(); x++) gradient_batch_sum -= (*it).batch_gradients[x];
      float rolling_gradient = gradient_batch_sum * (*it).last_gradient_batch_sum;
      (*it).last_gradient_batch_sum = gradient_batch_sum;
      (*it).batch_gradients.clear();
      (*it).weight = (*it).next_weight;
      if (rolling_gradient > 0) {
        float update_val = std::min( (*it).last_update_val * kResilientPropFaster, kResilientPropDeltaMax); 
        (*it).last_update_val = (*it).update_val;
        (*it).update_val = update_val;
        (*it).last_weight_delta = (*it).weight_delta;
        (*it).weight_delta = -sgn(gradient_batch_sum) * (*it).update_val;
        (*it).next_weight = (*it).weight + (*it).weight_delta;
      } else if (rolling_gradient < 0) {
        (*it).update_val = std::max( (*it).last_update_val * kResilientPropSlower, kResilientPropUpdateMin);
        (*it).next_weight = (*it).weight -  (*it).last_weight_delta;
        (*it).last_gradient_batch_sum = 0;
      } else {
        (*it).last_weight_delta = (*it).weight_delta;
        (*it).weight_delta = -sgn(gradient_batch_sum) * (*it).update_val;
        (*it).next_weight = (*it).weight + (*it).weight_delta;
      }
      for (std::vector<synapse_t>::iterator it2 = (*it).child->parents().begin(); it2 != (*it).child->parents().end(); ++it2) {
        if((*it2).parent == this && (*it2).child == (*it).child) (*it2).weight = (*it).weight;
      }
    }
  }
}

} //namespace neuralplex