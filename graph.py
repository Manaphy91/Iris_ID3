from graphviz import Digraph
import tree

def create_graph(root, graph_name, show_prob=False, format='svg', view=True):
    if type(root) != type(tree.Node('', [])):
        print("Error: root argument {} is not of type tree.Node!")
    else:
        filename = "{}.{}".format(graph_name, format)
        graph = Digraph(graph_name, filename=filename, format=format)
        graph.attr('node', shape='box')

        __create_graph(root, graph, show_prob)

        if view: 
            graph.view() 
        else:
            graph.save()

        return graph

def __create_graph(node, graph, show_prob):
    '''Helper function that from a node of type tree.Node, a graph of type
    Digraph and a boolean named show_prob return a graph of the id3 decision
    tree stored in node.
    '''
    treshold_value = node.get_treshold_value()
    treshold_name = node.get_treshold_name()
    node_hash = str(hash(node))

    # if node is a leaf, then show its confidence
    if len(node.get_sons()) == 0:
        if show_prob:
            value = "{}\n({} %)".format(node.get_value(), node.get_prob())
        else:
            value = "{}".format(node.get_value())
    else:
    # otherwise put treshold_name as node's label and treshold_value in
    # inequalities as edges' labels
        value = str(treshold_name)

        # call __create_graph on each son of this node
        sons = node.get_sons()

        graph.edge(node_hash, str(hash(sons[0])), \
            label="<{}".format(treshold_value))
        __create_graph(sons[0], graph, show_prob)

        graph.edge(node_hash, str(hash(sons[1])), \
            label=">={}".format(treshold_value))
        __create_graph(sons[1], graph, show_prob)

    graph.node(node_hash, value)
