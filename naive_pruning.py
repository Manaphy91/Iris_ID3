import tree
import performances as perf
import id3lib as id3
import copy

def is_leaf(node):
    return len(node.get_sons()) == 0

def prune_tree(root, test_mat, test_res, res_lst):
    __prune_tree(root, test_mat, test_res, res_lst, root)

def __prune_tree(root, test_mat, test_res, res_lst, node_ptr):
    if is_leaf(node_ptr):
        return
    else:
        for son in node_ptr.get_sons():
            __prune_tree(root, test_mat, test_res, res_lst, son)
    
        conf_mat_1 = perf.calculate_confusion_matrix(root, test_mat, test_res, \
            res_lst) 

        stats = node_ptr.get_stats()
        value = id3.get_greater(stats)
        total = sum(v for _, v in node_ptr.get_stats().items())
        prob = round(stats[value] / total, 3)
        
        parent = node_ptr.get_parent()

        if parent == None:
            return

        old_sons = parent.get_sons()

        sons = []
        for son in parent.get_sons():
            if son != node_ptr:
                sons.append(son)
        sons.append(tree.Node(value, [], stats=total))

        parent.set_sons(sons)

        conf_mat_2 = perf.calculate_confusion_matrix(root, test_mat, test_res, \
            res_lst) 

        if perf.get_accuracy(conf_mat_1) > perf.get_accuracy(conf_mat_2):
            parent.set_sons(old_sons)
