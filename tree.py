class Node:
    
    def __init__(self, value, sons, prob=100.0):
        self._value = value
        self._sons = sons
        self._prob = prob 

    def set_value(self, value):
        self._value = value

    def set_sons(self, sons):
        self._sons = sons

    def set_prob(self, prob):
        self._prob = prob

    def get_value(self):
        return self._value

    def get_sons(self):
        return self._sons

    def get_prob(self):
        return self._prob 

    def __str__(self):
        str = ""
        if callable(self._value):
            if self._value.__doc__ != None:
                str += self._value.__doc__
            else:
                str += "lambda function"
        else:
            str += self._value
        return "[Value: {}, prob: {}, sons: {}]".format(str, self._prob, \
            self._sons)

