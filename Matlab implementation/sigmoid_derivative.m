% the derivative of an neuron output
%Given an output value from a neuron, we need to calculate it’s slope.
function td= transfer_derivative(output)
	td= output * (1.0 - output);