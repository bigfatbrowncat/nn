#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include <vector>

using namespace std;

int width, height;

struct connection;

struct neuron {
	double value;
	double value_new;

	double error;

	bool value_is_expected;
	double expected_value;

	vector<connection*> input_connections;
	vector<connection*> output_connections;

	double const_weight;
};

struct connection {
	neuron* left;
	neuron* right;
	double weight;
};

struct neuron_net {
	vector<neuron*> neurons;
	vector<connection*> connections;

	vector<int> layer_heights;
	vector<vector<neuron>::size_type> layer_starts;
	int layers;

	// Allocates classic Rosenblatt perceptron
	static neuron_net* alloc_classic(int layers, const vector<int>& layer_heights) {

		neuron_net* pres = new neuron_net();

		neuron_net& res = *pres;
		res.layers = layers;
		res.layer_heights = layer_heights;

		for (int l = 0; l < layers; l++) {

			vector<neuron>::size_type layer_start = res.neurons.size();
			res.layer_starts.push_back(layer_start);

			for (int i = 0; i < layer_heights[l]; i++) {

				res.neurons.push_back(new neuron());
				neuron* index = res.neurons.back();
				neuron& n = *index;

				n.value = (double) rand() / RAND_MAX - 0.5;
				n.const_weight = (double) rand() / RAND_MAX - 0.5;
				n.error = 0;

				n.value_is_expected = false;
				n.expected_value = 0;

				if (l > 0) {
					n.input_connections.reserve(layer_heights[l - 1]);

					vector<neuron>::size_type prev_layer_start =
							*(res.layer_starts.end() - 2);

					for (int j = 0; j < layer_heights[l - 1]; j++) {
						neuron* index_prev = res.neurons[prev_layer_start + j];

						res.connections.push_back(new connection());
						connection* con_index = res.connections.back();
						connection& con = *con_index;

						con.left = index_prev;
						con.right = index;
						con.weight = (double) rand() / RAND_MAX - 0.5;

						n.input_connections.push_back(con_index);
						index_prev->output_connections.push_back(con_index);
					}

				}

				if (l < layers - 1) {
					n.output_connections.reserve(layer_heights[l + 1]);
				} else {
					n.output_connections.reserve(1);
				}

			}
		}

		return pres;
	}
protected:
	neuron_net() {
	}

private:
	neuron_net& operator =(const neuron_net& other);
	neuron_net(const neuron_net& other);
};

// s = 2 / (1+e^(-2x)) - 1
double sigmoid(double x) {
	return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

// s' = (1 + s)(1 - s)
double sigmoid_diff(double x) {
	double s = sigmoid(x);
	return 1.0 - s * s;
}

// Calculates a forward step for a single neuron
void net_neuron_calc_new_value_step(neuron* neuron) {
	if (neuron->input_connections.size() > 0) {
		double weighted_sum = neuron->const_weight * 1.0;
		for (size_t i = 0; i < neuron->input_connections.size(); i++) {
			weighted_sum += neuron->input_connections[i]->weight
					* neuron->input_connections[i]->left->value;
		}
		neuron->value = sigmoid(weighted_sum);
	}
}

// Calculates a backward step for a single neuron
void net_neuron_calc_error_step(neuron* neuron) {
	double weighted_sum = 0;
	if (neuron->value_is_expected) {
		weighted_sum = neuron->expected_value - neuron->value;
	} else {
		weighted_sum = 0;
		for (size_t i = 0; i < neuron->output_connections.size(); i++) {
			weighted_sum += neuron->output_connections[i]->weight
					* neuron->output_connections[i]->right->error;
		}
	}

	neuron->error = sigmoid_diff(neuron->value) * weighted_sum;
}

// Calculates single neuron weights
void net_neuron_calc_new_weights_step(neuron* neuron) {
	double alpha = 0.0001;

	neuron->const_weight = neuron->const_weight + alpha * neuron->error * 1.0;

	for (size_t i = 0; i < neuron->input_connections.size(); i++) {
		neuron->input_connections[i]->weight =
				neuron->input_connections[i]->weight
						+ alpha * neuron->error
								* neuron->input_connections[i]->left->value;
	}
}

// Calculates a value for the whole net
void net_calc_value_step(neuron_net* net) {
	for (int k = net->neurons.size() - 1; k >= 0; k--) {
		net_neuron_calc_new_value_step(net->neurons[k]);
	}
}

void net_calc_error_step(neuron_net* net) {
	for (unsigned int k = 0; k < net->neurons.size(); k++) {
		net_neuron_calc_error_step(net->neurons[k]);
	}

	for (int k = net->neurons.size() - 1; k >= 0; k--) {
		net_neuron_calc_new_weights_step(net->neurons[k]);
	}
}

void test_xor() {

	// Topology
	vector<int> layer_heights = { 2, 2, 1 };

	neuron_net* nn = neuron_net::alloc_classic(3, layer_heights);

	for (unsigned int i = 0; i < nn->neurons.size(); i++) {
		nn->neurons[i]->value_is_expected = false;
	}

	int last = nn->neurons.size() - 1;
	nn->neurons[last]->value_is_expected = true;

	// Learning
	int inputA[] = { 0, 0, 1, 1 };
	int inputB[] = { 0, 1, 0, 1 };
	int expectedOut[] = { 0, 1, 1, 0 };

	for (int s = 0; s < 300000; s++) {
		nn->neurons[0]->value = inputA[s % 4];
		nn->neurons[1]->value = inputB[s % 4];
		nn->neurons[last]->expected_value = expectedOut[s % 4];
		for (int k = 0; k < 15; k++) {
			net_calc_value_step(nn);
			net_calc_error_step(nn);
		}
	}

	// Testing
	double delta = 0;
	for (int i = 0; i < 4; i++) {
		nn->neurons[0]->value = inputA[i];
		nn->neurons[1]->value = inputB[i];
		for (int s = 0; s < nn->layers; s++) {
			net_calc_value_step(nn);
		}

		delta += abs(nn->neurons[last]->value - expectedOut[i]);
	}
	printf("test_xor delta %f\n", delta);

	delete nn;
}

int main(int argc, char** argv) {

	test_xor();

	return 0;
}
