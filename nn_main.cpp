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
	neuron* neurons;
	int count;
    
    neuron* input_neurons;
    int input_count;

    int* layer_heights;
    int layers;
};

// s = 2 / (1+e^(-2x)) - 1
double sigmoid(double x) {
	return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

// s' = (1 + s)(1 - s)
double sigmoid_diff(double x) {
	double s = sigmoid(x);
	return 1.0 - s*s;
}

// Allocates classic Rosenblatt perceptron
neuron_net net_alloc_classic(int layers, int layer_heights[]) {
	
	// Allocating memory
	// Calculating neurons count
	int count = 0;
    int layer_index_starts[layers + 1];
    layer_index_starts[0] = 0;
    
	for (int k = 0; k < layers; k++) {
		count += layer_heights[k];
        layer_index_starts[k + 1] = count;
	}
	
	neuron* net = new neuron[count];
    
	for (int l = 0; l < layers; l++) {
		for (int i = 0; i < layer_heights[l]; i++) {
            int index = layer_index_starts[l] + i;
			
			//net[index].input_connections_count = 0;
			//net[index].output_connections_count = 0;

			net[index].value = (double)rand() / RAND_MAX - 0.5;
			net[index].const_weight = (double)rand() / RAND_MAX - 0.5;
			net[index].error = 0;

			net[index].value_is_expected = false;
			net[index].expected_value = 0;
			
			if (l > 0) {
				net[index].input_connections.reserve(layer_heights[l - 1]);
				
				for (int j = 0; j < layer_heights[l - 1]; j++) {
                    int index_prev = layer_index_starts[l - 1] + j;

					connection* con = new connection();
				
					con->left = &net[index_prev];
					con->right = &net[index];
					con->weight = (double)rand() / RAND_MAX - 0.5;
				
					net[index].input_connections.push_back(con); // [net[index].input_connections_count++] = con;
					net[index_prev].output_connections.push_back(con); // [net[index_prev].output_connections_count++] = con;
				}
                
			}
			
			if (l < layers - 1) {
				net[index].output_connections.reserve(layer_heights[l + 1]);
			} else {
				net[index].output_connections.reserve(1);
			}
						
			index ++;
		}
	}
    
	neuron_net res;
    res.layers = layers;
    res.layer_heights = new int[layers];
    for (int l = 0; l < layers; l++) {
        res.layer_heights[l] = layer_heights[l];
    }
	res.neurons = net;
	res.count = count;
	
	return res;
}

// Calculates a forward step for a single neuron
void net_neuron_calc_new_value_step(neuron* neuron) {
    if (neuron->input_connections.size() > 0) {
        double weighted_sum = neuron->const_weight * 1.0;
        for (size_t i = 0; i < neuron->input_connections.size(); i++) {
            weighted_sum += neuron->input_connections[i]->weight * neuron->input_connections[i]->left->value;
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
            weighted_sum += neuron->output_connections[i]->weight * neuron->output_connections[i]->right->error;
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
				neuron->input_connections[i]->weight +
				alpha * neuron->error * neuron->input_connections[i]->left->value;
	}
}

// Calculates a value for the whole net
void net_calc_value_step(neuron_net net) {
	for (int k = net.count - 1; k >= 0; k--) {
		net_neuron_calc_new_value_step(&net.neurons[k]);
    }
}

void net_calc_error_step(neuron_net net) {
	for (int k = 0; k < net.count; k++) {
		net_neuron_calc_error_step(&net.neurons[k]);
	}

	for (int k = net.count - 1; k >= 0; k--) {
		net_neuron_calc_new_weights_step(&net.neurons[k]);
	}
}

void print(neuron_net nn) {
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			nn.neurons[0].value = i;
			nn.neurons[1].value = j;

			// Calculating the net
			for (int s = 0; s < nn.layers; s++) { net_calc_value_step(nn); }

			printf("%d xor %d = %lf\n", i, j, nn.neurons[nn.count - 1].value);
		}
	}
	printf("\n");
}

void test_xor() {

	// Topology
	int layer_heights[] = { 2, 2, 1 };

	neuron_net nn = net_alloc_classic(3, layer_heights);
    
    for (int i = 0; i < nn.count; i++) {
        //nn.neurons[i].value = 0;
    	nn.neurons[i].value_is_expected = false;
    }

	int last = nn.count - 1;
	nn.neurons[last].value_is_expected = true;

	// Learning
	int inputA[] =      { 0, 0, 1, 1 };
	int inputB[] =      { 0, 1, 0, 1 };
	int expectedOut[] = { 0, 1, 1, 0 };

	for (int s = 0; s < 300000; s++) {
		nn.neurons[0].value = inputA[s % 4];
		nn.neurons[1].value = inputB[s % 4];
		nn.neurons[last].expected_value = expectedOut[s % 4];
		for (int k = 0; k < 15; k++) {
			net_calc_value_step(nn);
			net_calc_error_step(nn);
		}
	}

	// Testing
	double delta = 0;
	for (int i = 0; i < 4; i ++) {
		nn.neurons[0].value = inputA[i];
		nn.neurons[1].value = inputB[i];
		for (int s = 0; s < nn.layers; s++) { net_calc_value_step(nn); }

		delta += abs(nn.neurons[last].value - expectedOut[i]);
	}
	printf("test_xor delta %f\n", delta);
}

int main(int argc, char** argv) {

	test_xor();
}
