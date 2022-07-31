function swishd=swish_derivative(output)
    num=(1+exp(-output))+(output*exp(-output));
    den=(1+exp(-output)).^2;
    swishd=num/den;