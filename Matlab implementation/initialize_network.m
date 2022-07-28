%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Below is a function named initialize_network() that creates a new neural
%network ready for training. It accepts three parameters, the number of
%inputs, the number of neurons to have in the hidden layer and the number
%of outputs. The hidden layer we create n_hidden
%neurons and each neuron in the hidden layer has n_inputs + 1 weights, one
%for each input column in a dataset and an additional one for the bias.
%The output layer that connects to the hidden layer has
%n_outputs neurons, each with n_hidden + 1 weights. This means that each
%neuron in the output layer connects to (has a weight for) each neuron in
%the hidden layer.



function network= initialize_network(n_inputs, n_hidden, n_outputs)
	net = {};
    hidden_layer={};
    output_layer={};
    
    for k=1:n_hidden;
        hid=[];
        for i=1:n_inputs+1;
            h=[(1-0).*rand(1, 'double')];
            hid=[hid h];
            s.weights=hid
        end
        hidden_layer=[hidden_layer,s];
        display(hidden_layer)
        
    end
	net.hidden=hidden_layer;
    for i=1:n_outputs;
        out=[];
        for k=1:n_hidden+1;
            o=[(1-0).*rand(1, 'double')];
            out=[out, o];
            s.weights=out;
        end
        %display(out
        output_layer=[output_layer, s];
        display(output_layer)
    end
    
    net.output=output_layer;
    network=net;