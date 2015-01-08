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

#ifndef NEURON_H_
#define NEURON_H_

#include<stdlib.h>
#include<string>
#include<vector>

namespace neuralplex {

// The neural network will start by neurons sorted input to output, Forward method is ran on each neuron,
// in a forward feeding manner to calculate the output values based on summing the parent neurons
// output multiplied by the weight between child and parent. We can then compare the ideal provided in
// the training step to the last neuon's output value to get an initial error value. The idea is to derive the gradient
// slope between the actual output value and the ideal output value and thus minimize the amount of error.
// We generate the gradient with the provided activation function and then spread the value backwards 
// through the network. This is done by reversing the order and  then calling the Backward method, once per
// node, from output to input. A bias neuron, a neuron whos output is always 1.0f, connected to each hidden
// neuron, and a second bias neuron connected to each output neuron is used to allow the network to arrive at
// outputs greater or smaller than the range if the provided activation function providing more signal coverage
// and likelihood to converge.
class Neuron {
 public:
  typedef struct {
    Neuron* parent;
    Neuron* child;
    float weight;
    float last_delta;
    float last_gradient_batch_sum;
    float update_val;
    float last_update_val;
    float next_weight;
    float weight_delta;
    float last_weight_delta;
    std::vector <float> batch_gradients;
  } synapse_t;

  Neuron(std::string name, float (*activation)(float), float (*activation_prime)(float));
  virtual ~Neuron();
  void ConnectTo(Neuron *n, float weight);
  void Forward();
  void Backward();
  void Learn(int learning_algo);
  float input() const { return input_; }
  void set_input(float input) { has_input_ = true; input_ = input; }
  float ideal() const { return ideal_; }
  void set_ideal(float ideal) { has_ideal_ = true; ideal_ = ideal; }
  std::string name() const { return name_; }
  float output() const { return output_; }
  float error() const { return error_; }
  float delta() const { return delta_; }
  int layer_idx() const { return layer_idx_; }
  void set_layer_idx(int layer_idx) { layer_idx_ = layer_idx; }
  std::vector <synapse_t>& parents()  { return parents_; }
  std::vector <synapse_t>& children()  { return children_; }

 private:
  void LearnBackProp();
  void LearnRProp();
  bool has_input_;
  bool has_ideal_;
  float summation_;
  float ideal_;
  float input_;
  std::string name_;
  float (*activation_)(float);
  float (*activation_prime_)(float);
  float output_;
  float error_;
  float delta_;
  int layer_idx_;
  std::vector <synapse_t> children_;
  std::vector <synapse_t> parents_;
};

}  //namespace neuralplex
#endif /*NEURON_H_*/

