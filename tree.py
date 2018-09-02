class Node:
    
    def __init__(self, value, sons, parent=None, prob=100.0, \
        treshold_name=None, treshold_value=None, approx_value=None, errors=0):
        self.__value = value
        self.__sons = sons
        self.__parent = parent
        self.__prob = prob
        self.__treshold_name = treshold_name
        self.__treshold_value = treshold_value
        self.__approx_value = approx_value
        self.__errors = errors

    def set_value(self, value):
        self.__value = value

    def set_sons(self, sons):
        self.__sons = sons

    def set_parent(self, parent):
        self.__parent = parent

    def set_prob(self, prob):
        self.__prob = prob

    def set_treshold_name(self, name):
        self.__treshold_name = name

    def set_treshold_value(self, value):
        self.__treshold_value = value

    def set_approx_value(self, approx_value):
        self.__approx_value = approx_value

    def add_errors(self):
        self.__errors += 1

    def get_value(self):
        return self.__value

    def get_sons(self):
        return self.__sons

    def get_parent(self):
        return self.__parent

    def get_prob(self):
        return self.__prob

    def get_treshold_name(self):
        return self.__treshold_name

    def get_treshold_value(self):
        return self.__treshold_value

    def get_approx_value(self):
        return self.__approx_value

    def get_errors(self):
        return self.__errors

    def __str__(self):
        str = ""
        if callable(self._value):
            if self.__value.__doc__ != None:
                str += self.__value.__doc__
            else:
                str += "lambda function"
        else:
            str += self.__value
        return "[value: {}, sons: {}, parent: {}, prob: {}, \
            treshold_name: {}, treshold_value: {}, approx_value: {}, \
            errors: {}]".format(str, self.__sons, self.__prob, \
            self.__treshold_name, self.__approx_value, self.__treshold_name, \
            self.__errors)
