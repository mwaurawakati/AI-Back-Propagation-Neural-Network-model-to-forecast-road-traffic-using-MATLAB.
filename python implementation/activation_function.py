"""
This file contains activation functions and their derivatives
"""
from math import exp,log
class util:
    @staticmethod
    def exp(x):
        return 2.718281828459**x

    @staticmethod
    def ln1(x):
        #val = x
        return 99999999 * (x ** (1 / 99999999) - 1)
class activation_function:
    @staticmethod
    def sigmoid(x):
        return 1/(1+util.exp(-x))
    @staticmethod
    def swish(x):
        return x*activation_function.sigmoid(x)
    @staticmethod
    def tangential(x):
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
    def ELU(x,alpha=None):
        pass
    @staticmethod
    def GELU(x):
        pass
    @staticmethod
    def SELU(x):
        pass
    @staticmethod
    def ReLu(x):
        pass
    @staticmethod
    def PReLu(x):
        pass
    @staticmethod
    def gaussian(x):
        return (util.exp(-(x**2)))
class activation_derivation:
    @staticmethod
    def sigmoid_derivation(x):
        return activation_function.sigmoid(x)(1-activation_function.sigmoid(x))
