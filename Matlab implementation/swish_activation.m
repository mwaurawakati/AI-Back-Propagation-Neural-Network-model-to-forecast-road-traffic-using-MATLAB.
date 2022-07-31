%https://en.wikipedia.org/wiki/Swish_function
%https://paperswithcode.com/method/swish#:~:text=Swish%20is%20an%20activation%20function,%22Swish%2D1%22).
function sig=swish_activation(x)
sig=(1+exp(-x))*x;