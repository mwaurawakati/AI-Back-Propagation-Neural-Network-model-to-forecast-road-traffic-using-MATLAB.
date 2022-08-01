"""
This file contains activation functions and their derivatives
"""
from math import exp,log
import tensorflow as tf
class util:
    @staticmethod
    def cosh(x):
        return (util.exp(x)+util.exp(-x))/2

    @staticmethod
    def sech(x):
        return util.cosh(x) ** (-1)

    @staticmethod
    def exp(x):
        return 2.718281828459**x

    @staticmethod
    def ln1(x):
        #val = x
        return 99999999 * (x ** (1 / 99999999) - 1)
    @staticmethod
    def erf(x):
        # save the sign of x
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * util.exp(-x * x)
        return sign * y  # erf(-x) = -erf(x)
    @staticmethod
    def sqrt(x):
        return x**(1/2)

class activation_function:
    @staticmethod
    def sigmoid(x):
        return 1/(1+util.exp(-x))

    @staticmethod
    def swish(x):
        return x*activation_function.sigmoid(x)

    @staticmethod
    def tanh(x):
        num=util.exp(x)-util.exp(-x)
        den=util.exp(x)+util.exp(-x)
        return num/den

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def binary_step(x):
        if x<0:
            return 0
        else:
            return 1

    @staticmethod
    def softplus(x):
        return util.ln(1+util.exp(x))

    @staticmethod
    def ELU(x,alpha=1):
        if x>0:
            return x
        else:
            return alpha*(util.exp(x)-1)

    @staticmethod
    def SELU(x):
        glambda=1.0507009873554804934193349852946
        galpha=1.6732632423543772848170429916717
        if x<0:
            return glambda*galpha*(util.exp(x)-1)
        else:
            return x

    @staticmethod
    def GELU(x):
        cdf = 0.5 * (1.0 + util.erf(x / util.sqrt(2.0)))
        return x * cdf

    @staticmethod
    def ReLu(x):
        return max(0.0, x)

    @staticmethod
    def PReLu(x,alpha=1):
        if x<0:
            return alpha*x
        else:
            return x


    @staticmethod
    def gaussian(x):
        return (util.exp(-(x**2)))

    @staticmethod
    def lReLu(x):
        if x<0:
            return 0.01*x
        else:
            return x

class activation_derivation:

    """
    This are the derivatives of activation functions that can be used for back propagation
    """
    @staticmethod
    def sigmoid_derivation(x):
        return activation_function.sigmoid(x)(1-activation_function.sigmoid(x))

    @staticmethod
    def tanh_derivative(x):
        return 1-(activation_function.tanh(x))**2

    @staticmethod
    def gaussian_derivative(x):
        return -2*x*activation_function.gaussian(x)

    @staticmethod
    def swish_derivative(x):
        num=1+util.exp(-x)+(x*util.exp(-x))
        den=(1+util.exp(-x))**2
        return num/den
    @staticmethod
    def derivative_softplus(x):
        return activation_function.sigmoid(x)

    @staticmethod
    def derivative_GELU(x):
        return 0.5*activation_function.tanh(0.0356774*x**3+0.797885*x)+\
               ((0.0535161*x**3+0.398942*x)*(util.sech(0.0356774*x**3+0.797885*x))**2)+0.5

