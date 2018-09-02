import tree
import id3lib as id3

def make_stats(tree, attr_lst, res):
    approx_value = tree.get_approx_value()
    if approx_value != res:
        tree.add_errors()

    func = tree.get_value()
    if callable(func):
        if func(attr_lst):
            make_stats(tree.get_sons()[0], attr_lst, res)
        else:
            make_stats(tree.get_sons()[1], attr_lst, res)
