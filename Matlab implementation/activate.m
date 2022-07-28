% Calculate neuron activation for an input
%Neuron activation is calculated as the weighted sum of the inputs.
function act= activate(weights, inputs)
	activation = weights(length(weights))
	for i=1:(length(weights)-1)
		activationi = weights(i) * inputs(i)
        activation = activation+activationi
    end
	act= activation