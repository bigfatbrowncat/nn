#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include <vector>
#include <memory>

using namespace std;

struct neuron;
struct connection;

// s = 2 / (1+e^(-2x)) - 1
double sigmoid(double x) {
	return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

// s' = (1 + s)(1 - s)
double sigmoid_diff(double x) {
	double s = sigmoid(x);
	return 1.0 - s * s;
}

struct connection {
	shared_ptr<neuron> left;
	shared_ptr<neuron> right;
	double weight;
};

struct neuron {
	double value;
	double value_new;

	double error;

	bool value_is_expected;
	double expected_value;

	vector<shared_ptr<connection>> input_connections;
	vector<shared_ptr<connection>> output_connections;

	double const_weight;

	// Calculates a forward step for a single neuron
	void calc_new_value() {
		if (input_connections.size() > 0) {
			double weighted_sum = const_weight * 1.0;
			for (size_t i = 0; i < input_connections.size(); i++) {
				weighted_sum += input_connections[i]->weight
						* input_connections[i]->left->value;
			}
			value = sigmoid(weighted_sum);
		}
	}

	// Calculates a backward step for a single neuron
	void calc_error() {
		double weighted_sum = 0;
		if (value_is_expected) {
			weighted_sum = expected_value - value;
		} else {
			weighted_sum = 0;
			for (size_t i = 0; i < output_connections.size(); i++) {
				weighted_sum += output_connections[i]->weight
						* output_connections[i]->right->error;
			}
		}

		error = sigmoid_diff(value) * weighted_sum;
	}

	// Calculates single neuron weights
	void calc_new_weights() {
		double alpha = 0.01;

		const_weight = const_weight + alpha * error * 1.0;

		for (size_t i = 0; i < input_connections.size(); i++) {
			input_connections[i]->weight =
					input_connections[i]->weight
					+ alpha * error * input_connections[i]->left->value;
		}
	}

};

struct layer {
	vector<shared_ptr<neuron>> neurons;
};

struct neuron_net {
	vector<shared_ptr<layer>> layers;
	vector<shared_ptr<connection>> connections;

	// Allocates classic Rosenblatt perceptron
	static shared_ptr<neuron_net> alloc_classic(int layers, const vector<int>& layer_heights) {

		auto pres = shared_ptr<neuron_net>(new neuron_net());
		neuron_net& res = *pres;

		// Creating new neurons
		for (auto height : layer_heights) {
			auto new_layer = shared_ptr<layer>(new layer());
			for (int i = 0; i < height; i++) {
				auto new_neuron = shared_ptr<neuron>(new neuron());
				new_layer->neurons.push_back(new_neuron);

				neuron& n = *new_neuron;
				n.value = (double) rand() / RAND_MAX - 0.5;
				n.const_weight = (double) rand() / RAND_MAX - 0.5;
				n.error = 0;

				n.value_is_expected = false;
				n.expected_value = 0;
			}
			res.layers.push_back(new_layer);
		}

		if (res.layers.size() >= 2) {
			for (auto layer_iter = res.layers.begin() + 1; layer_iter != res.layers.end(); layer_iter ++) {
				auto prev_layer_iter = layer_iter - 1;
				for (auto &n : (*layer_iter)->neurons) {
					for (auto &prev_n : (*prev_layer_iter)->neurons) {
						auto new_connection = shared_ptr<connection>(new connection());
						res.connections.push_back(new_connection);

						new_connection->left = prev_n;
						new_connection->right = n;
						new_connection->weight = (double) rand() / RAND_MAX - 0.5;

						prev_n->output_connections.push_back(new_connection);
						n->input_connections.push_back(new_connection);
					}
				}
			}
		}

		return pres;
	}

	// Calculates a value for the whole net
	void calc_values() {
		for (auto &layer : layers) {
			for (auto &neuron : layer->neurons) {
				neuron->calc_new_value();
			}
		}
	}

	void calc_errors() {
		for (auto layer = layers.rbegin(); layer != layers.rend(); layer++) {
			for (auto &neuron : (*layer)->neurons) {
				neuron->calc_error();
				neuron->calc_new_weights();
			}
		}
	}
protected:
	neuron_net() {
	}

private:
	neuron_net& operator =(const neuron_net& other);
	neuron_net(const neuron_net& other);
};

void test_xor() {

	// Topology
	vector<int> layer_heights = { 2, 2, 1 };

	shared_ptr<neuron_net> nn = shared_ptr<neuron_net>(neuron_net::alloc_classic(3, layer_heights));

	auto last_layer = nn->layers.back();

	for (auto &neuron : last_layer->neurons) {
		neuron->value_is_expected = true;
	}

	// Learning
	int inputA[] = { 0, 0, 1, 1 };
	int inputB[] = { 0, 1, 0, 1 };
	int expectedOut[] = { 0, 1, 1, 0 };

	for (int s = 0; s < 300000; s++) {
		nn->layers.front()->neurons[0]->value = inputA[s % 4];
		nn->layers.front()->neurons[1]->value = inputB[s % 4];
		nn->layers.back()->neurons[0]->expected_value = expectedOut[s % 4];

		nn->calc_values();
		nn->calc_errors();
	}

	// Testing
	double delta = 0;
	for (int i = 0; i < 4; i++) {
		nn->layers.front()->neurons[0]->value = inputA[i];
		nn->layers.front()->neurons[1]->value = inputB[i];

		nn->calc_values();

		delta += abs(nn->layers.back()->neurons[0]->value - expectedOut[i]);
	}
	printf("test_xor delta %f\n", delta);
}

int main(int argc, char** argv) {

	test_xor();

	return 0;
}
