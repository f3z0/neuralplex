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

#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include<stdlib.h>
#include<string>
#include<vector>
#include "Neuron.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

// NeuralNet class is responsible for the organization of neurons and facilitating intersynaptic communication.
// It also provides the external interface so that host system can generate a function approximation using only
// partial or incomplete training data sets. At the present time, neuralplex is a 3 layer network, which is to say
// that there is an input layer, a hidden layer and an output layer. There is no upper limit on the number of
// neurons per level which are configured with n_input, n_hidden and n_output respectively. To use this class
// just initialize with number neurons per layer, an activation function like sigmoid or tanh and the derivative  of
// the activation function. You then call Train (once only) providing your training data set and neuralplex learns.
// If convergence was achieved, which you can check by ensuring the global error return from calling Train is
// smaller or equal to kNeuralLearningThreshold. If so, the compute method is now an approximation of
// training data function. If convergence fails, you should try to tweak the number neurons, the training data,
// the learning type (back prop and risilient prop currently supported) and potentially values in
// neural_net_constants.h. 
namespace neuralplex {

class NeuralNet {
 public:
  // Used to sort the network with the input layers first so that each layer can fully complete delta and gradient
  // calculations prior to calculating the next layer.
  struct ForwardPropagation {
    bool operator()( const Neuron* a, const Neuron* b ) const {
      return a->layer_idx() < b->layer_idx();
    }
  };
  // Similar to ForwardPropogation but flipped for stepping through the backwards propagation phase.
  struct BackPropagation {
    bool operator()( const Neuron* a, const Neuron* b ) const {
      return a->layer_idx() > b->layer_idx();
    }
  };
  
  //NeuralNet(): construct a new NeuralNet
  // n_input: number of neurons in input layer
  // n_hidden: number of neurons in hidden layer
  // n_output: number of neurons in output layer 
  // activation: the activation function used for node delta calculation
  // activation_p: the derivative of the activation function used for gradient decent
  // start_weights: optional if you wish to provide your own random or pre-trained starting weight values.
  NeuralNet (int n_input, int n_hidden, int n_output, float (*activation)(float), float (*activation_p)(float));
  NeuralNet (int n_input, int n_hidden, int n_output, float (*activation)(float), float (*activation_p)(float), float *start_weights);
  virtual ~NeuralNet();
  // training_data: inputs followed by ideal outputs per row, rows are joined to form a 1d array of training_data.
  // batch_size: number of input+output pairs in training data
  // learning_algo: kLearningAlgorithmsResilientProp and kLearningAlgorithmsBackProp currently supported.
  float Train(float training_data[], int batch_size,  int learning_algo);
  // inputs: array of approximated functions inputs
  // outputs: results of approximated function with supplied inputs
  void Compute(float inputs[], float* outputs);
  // Returns pretty formatted string JSON representation of the neural network in present state.
  const char * ToPrettyJSON() {
    rapidjson::StringBuffer *buffer = new rapidjson::StringBuffer();
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(*buffer);
    this->ToJSON(writer);
    return buffer->GetString();
  }
  // Returns string JSON representation of the neural network in present state.
  const char * ToJSON() {
    rapidjson::StringBuffer *buffer = new rapidjson::StringBuffer();
    rapidjson::Writer<rapidjson::StringBuffer> writer(*buffer);
    this->ToJSON(writer);
    return buffer->GetString();
  }
  // Streams to rapidjson writer, the JSON representation of the neural network in present state.
  template <typename Writer>
  void ToJSON(Writer& writer) const {
    writer.StartObject();
    writer.String(("inputNeurons"));
    writer.StartArray();
    for (std::vector<Neuron*>::const_iterator neuronItr = input_neurons_.begin(); neuronItr != input_neurons_.end(); ++neuronItr)
    (*neuronItr)->ToJSON(writer);
    writer.EndArray();
    writer.String(("biasNeurons"));
    writer.StartArray();
    for (std::vector<Neuron*>::const_iterator neuronItr = bias_neurons_.begin(); neuronItr != bias_neurons_.end(); ++neuronItr)
    (*neuronItr)->ToJSON(writer);
    writer.EndArray();
    writer.String(("hiddenNeurons"));
    writer.StartArray();
    for (std::vector<Neuron*>::const_iterator neuronItr = hidden_neurons_.begin(); neuronItr != hidden_neurons_.end(); ++neuronItr)
    (*neuronItr)->ToJSON(writer);
    writer.EndArray();
    writer.String(("outputNeurons"));
    writer.StartArray();
    for (std::vector<Neuron*>::const_iterator neuronItr = output_neurons_.begin(); neuronItr != output_neurons_.end(); ++neuronItr)
    (*neuronItr)->ToJSON(writer);
    writer.EndArray();
    writer.EndObject();
  }
  // this is the number of training iterations that were required to converge
  int epoch() const { return epoch_; }

private:
  void BuildNetwork(float (*activation)(float), float (*activation_p)(float), float *start_weights);
  void NormalizeInputs(float* training_data, int batch_size);
  std::vector<Neuron*> input_neurons_;
  std::vector<Neuron*> bias_neurons_;
  std::vector<Neuron*> hidden_neurons_;
  std::vector<Neuron*> output_neurons_;
  std::vector<Neuron*> neurons_;
  int n_input_;
  int n_hidden_;
  int n_output_;
  int epoch_;
  float max_float_training_;
  float min_float_training_;
};

} //namespace neuralplex
#endif /*NEURAL_NET_H_*/
