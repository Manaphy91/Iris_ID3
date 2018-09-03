import tree
import id3lib as id3

def is_leaf(node):
    return len(node.get_sons()) == 0

def make_stats(root, attr_lst, res):
    approx_value = root.get_approx_value()
    if approx_value != res:
        root.add_errors()

    func = root.get_value()
    if callable(func):
        if func(attr_lst):
            make_stats(root.get_sons()[0], attr_lst, res)
        else:
            make_stats(root.get_sons()[1], attr_lst, res)

def prune_tree(root):
    if is_leaf(root):
        return

    sons = root.get_sons()
    for son in sons:
        prune_tree(son)

    if is_leaf(sons[0]) and is_leaf(sons[1]):
        if (sons[0].get_errors() + sons[1].get_errors()) >= root.get_errors():
            root.set_sons([])
            root.set_value(root.get_approx_value())

def prune(tree, val_set, val_res):
    for i in range(len(val_res)):
        make_stats(tree, val_set[i,:], val_res[i])

    prune_tree(tree)
