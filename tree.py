class Node:
    
    def __init__(self, value, sons, prob=100.0, treshold_name=None, \
        treshold_value =None):
        self.__value = value
        self.__sons = sons
        self.__prob = prob
        self.__treshold_name = treshold_name
        self.__treshold_value = treshold_value

    def set_value(self, value):
        self.__value = value

    def set_sons(self, sons):
        self.__sons = sons

    def set_prob(self, prob):
        self.__prob = prob

    def set_treshold_name(self, name):
        self.__treshold_name = name

    def set_treshold_value(self, value):
        self.__treshold_value = value

    def get_value(self):
        return self.__value

    def get_sons(self):
        return self.__sons

    def get_prob(self):
        return self.__prob

    def get_treshold_name(self):
        return self.__treshold_name

    def get_treshold_value(self):
        return self.__treshold_value

    def __str__(self):
        str = ""
        if callable(self._value):
            if self.__value.__doc__ != None:
                str += self.__value.__doc__
            else:
                str += "lambda function"
        else:
            str += self.__value
        return "[value: {}, sons: {}, prob: {}, treshold_name: {}, \
            treshold_value: {}]".format(str, self.__sons, self.__prob, \
            self.__treshold_name, self.__treshold_name)
