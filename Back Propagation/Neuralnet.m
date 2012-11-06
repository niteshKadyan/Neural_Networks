%% Author : Nitesh Kumar IIIT Bangalore %%
%% Neural Network to learn xor function using sigmoidal functions %%
%% Initialize %%
inputs = [-1, 0, 0; -1, 0, 1; -1, 1, 0; -1, 1, 1];
outputs = [0; 1; 1; 0];

no_of_inputs = 3;
no_of_input_nodes = 2;
no_of_output_nodes = 1;

activation_layer_one = zeros(1, no_of_input_nodes);
activation_layer_two = zeros(1, no_of_output_nodes);

output_layer_one = zeros(1, no_of_input_nodes);
output_layer_two = zeros(1, no_of_output_nodes);

layer_one_weights = 1.0 * rand(no_of_inputs, no_of_input_nodes)
%layer_two_weights = rand(no_of_input_nodes + 1, no_of_output_nodes)
layer_two_weights = randi([-2.0, +2.0], [no_of_input_nodes + 1, no_of_output_nodes])

Error_layer_one = zeros(1, no_of_input_nodes);
Error_layer_two = zeros(1, no_of_output_nodes);

%% Train %%
iterations = 10000;
Error = zeros(1,10000);
it = 1:10000;
for i = 1:iterations;
    for j = 1:4;
        %Forward Propagation
        activation_layer_one = inputs(j, :) * layer_one_weights;
        output_layer_one = sigmoid(activation_layer_one);
        output_layer_one = [-1, output_layer_one];
        activation_layer_two = output_layer_one * layer_two_weights;
        output_layer_two = sigmoid(activation_layer_two);

        %Calculate Error
        Error_layer_two = (outputs(j) - output_layer_two)^2;
        dError_layer_two = -2 * (outputs(j) - output_layer_two) * output_layer_two * (1 - output_layer_two);
        Error_layer_one = layer_two_weights * dError_layer_two;
        Error_layer_one = removerows(Error_layer_one, [1]);
        Error_layer_one(1) = Error_layer_one(1) * output_layer_one(2) * (1 - output_layer_one(2));
        Error_layer_one(2) = Error_layer_one(2) * output_layer_one(3) * (1 - output_layer_one(3));
        
        %Backward Propagation
        layer_two_weights = layer_two_weights - 0.1 * output_layer_one' * dError_layer_two';
        layer_one_weights = layer_one_weights - 0.1 * inputs(j, :)' * Error_layer_one';
    end
    Error_layer_one;
    Error(i) = Error_layer_two;
end
plot(it, Error)
layer_two_weights;
layer_one_weights;

%% Test %%

for i = 1:4;
    activation_layer_one = inputs(i, :) * layer_one_weights;
    output_layer_one = sigmoid(activation_layer_one);
    activation_layer_two = [-1, output_layer_one] * layer_two_weights;
    output_layer_two = sigmoid(activation_layer_two)
end