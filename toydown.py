import numpy as np
import scipy as sp
from scipy.optimize import minimize
from treelib import Node, Tree
import dask

class GeoUnit(object):
    """ This class stores the data inside each Node in the tree.
    """
    def __init__(self, name, parent, attributes, identifier=None):
        """ Args:
                name         : Str, Name of the Node
                parent       : Str, The identifier of the parent of the Node
                attributes   : np.array(dtype=int), The array of population counts for
                               the Node.
                identifier   : Str, A string that is unique to the Node in the entire Tree
        """
        self.name = name
        self.parent = parent
        self.attributes = attributes
        self.identifier = identifier if identifier else name

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = [ "{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "%s(%s)" % (name, ", ".join(kwargs))


class ToyDown(Tree):
    def __init__(self, geounits, num_levels, eps_budget, eps_splits, parallel=False):
        """ Initializes the Tree and populates it.
            geounits   : List of GeoUnits that will form the nodes of the Tree.
            eps_budget : Float, Epsilon budget across all levels
            eps_splits : List denoting the % of splits in epsilon value by level
                         eg. if the hierarchy is [Country, State, County, District] then
                         the eps_splits could look like [0, 0.33, 0.33, 0.34]
        """
        super(ToyDown, self).__init__()
        self.eps_budget = eps_budget
        self.populate_tree(geounits)
        self.add_levels_to_node(self.get_node(self.root), 0)
        self.eps_values = self.epsilon_values(num_levels, eps_splits, eps_budget)
        self.parallel = parallel

    def epsilon_values(self, num_levels, eps_splits, eps_budget):
        """ Stores the epsilon values as a List of Floats.
            The eps_values list should be in decreasing order of size
            eg [Country, State, County, Tract, BlockGroup, Block]
        """
        assert(len(eps_splits) == num_levels)
        eps_values = []

        for fraction in eps_splits:
            eps_values.append(fraction * eps_budget)

        return eps_values

    def populate_tree(self, geounits):
        """ Populates the Tree from a list of GeoUnits.
        """
        for unit in geounits:
            if unit.parent:
                self.create_node(unit.name, unit.identifier, parent=unit.parent, data=unit)
            else:
                # root node
                self.create_node(unit.name, unit.identifier, data=unit)

## TODO:: To match precinct toy model

    def add_laplacian_noise(self, node, epsilon):
        """ Adds Laplacian noise of parameter 1/`epsilon` to Node `node`. If `epsilon` is 0, 
            adds no noise.
        """
        shape = node.data.attributes.shape
        if epsilon == 0:
            noise = np.full(shape, 0)
        else:
            noise = np.random.laplace(scale=1/epsilon, size=shape)

        node.data.noised = node.data.attributes + noise
        node.data.noise = noise
        node.data.noise_type = "laplacian"

    def add_levels_to_node(self, node, level):
        """ Recursive function that adds the `level` as an attribute
            to the `node`.
        """
        self.get_node(node.identifier).data.level = level
        for child in self.children(node.identifier):
            self.add_levels_to_node(child, level+1)

    def adjust_children(self, node, objective_fun, node_cons, bounds, parental_equality, maxiter, verbose):
        """ Adjusts the children to add up to the parent.
        """
        adj_par = node.data.adjusted
        children = self.children(node.identifier)
        num_attributes = node.data.attributes.shape[0]
        num_children = len(children)
        noised_children = np.append([], [child.data.noised for child in children])
        unnoised_children = np.append([], [child.data.attributes for child in children])
        
        bnds = [(0, None)]*(num_children*num_attributes) if bounds == "non-negative" else bounds(num_children)

        if parental_equality:
            cons_children = [{'type': 'eq', 'fun': lambda x: np.dot(adj_par - np.sum(x.reshape(num_children, num_attributes), axis=0),
                                                                    adj_par - np.sum(x.reshape(num_children, num_attributes), axis=0))}]
            if not node_cons:
                cons = cons_children
            else:
                cons = node_cons(num_children) + cons_children
        
        if verbose: print("Adjusting children of {}".format(node.data.name))
        adj = minimize(objective_fun(noised_children), unnoised_children, constraints=cons, 
                       bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
        adjusted_children = adj.x

        for i, adjusted_child in enumerate(np.split(adjusted_children, num_children)):
            children[i].data.adjusted = adjusted_child
            children[i].data.error = children[i].data.attributes - children[i].data.adjusted

    def noise_and_adjust(self, objective_fun="L1", node_cons=None, bounds="non-negative", 
                         parental_equality=True, maxiter=200, verbose=False):
        """ Noises each node in the Tree and adjusts them to add back to their parent.
            This function simply serves as a wrapper function to the recursive 
            __noise_and_adjust_children(), and is started at the root of the tree.

            objective_fun     : Function to minimize over.  If "L1" uses the L1 norm, otherwise
                                takes a function that takes a noised point, and returns a function that
                                takes a point x and returns a scalar distance between the two points.
            node_cons         : Function that takes the number of children, n, and returns a list of constraints 
                                for the readjustment of a Node is subject to. 
                                Has form [{"type": "eq/ineq", "fun": lambda x: }]
                                See scipy.optimize constraint specification for more details.
            bounds            : Function that takes the number of children, n, and returns a list of 
                                (min, max) pairs for each attribute in each of the children, defining 
                                the bounds on that attribute. Use None for one of min or max when there is 
                                no bound in that direction.
                                A value of "non-negative", flags that all counts should be > 0
            parental_equality : Boolean flag - adds constraint that for all attributes, the sum of the 
                                children's counts should be equal to that of the parent's counts.
            maxiter           : Option to pass along to scipy optimizer.  Defines the maximum number of
                                iterations the optimizer should preform.  Default value 200.
            verbose           : Boolean flag -- if True prints debug info.
        """

        root = self.get_node(self.root)
        num_attributes = root.data.attributes.shape[0]
        if objective_fun == "L1": objective_fun = lambda n: lambda x: sp.linalg.norm(x-n, ord=1)
        
        # noise tree
        self.__noise_tree(root)

        # adjust tree
        if self.parallel:
            adj_root = self.__adjusted_root(root, objective_fun, node_cons, bounds,
                                            parental_equality, maxiter, verbose)
            delay = self.__adjust_tree_async(root, adj_root, objective_fun, node_cons, bounds,
                                            parental_equality, maxiter, verbose)
            adjusted = delay.compute()
            self.__update_adjusted_and_error(adjusted)

        else:
            self.__adjust_tree(root, objective_fun, node_cons, bounds,
                                             parental_equality, maxiter, verbose)

    def __noise_tree(self, node):
        self.add_laplacian_noise(node, self.eps_values[node.data.level])
        
        if node.is_leaf():
            return
        
        for child in self.children(node.identifier):
            self.__noise_tree(child)

    def __adjust_tree(self, node, objective_fun, node_cons, bounds, 
                                    parental_equality, maxiter, verbose):
        """ Recursively noises children and then "adjusts" the children to sum
            up to the population of the parent.
        """
        if node.is_leaf():
            return

        elif node.is_root():
            # Adjust root
            bnds = [(0, None)]*(node.data.attributes.shape[0]) if bounds == "non-negative" else bounds(1)
            
            cons = node_cons if not node_cons else node_cons(1)
            if verbose: print("Adjusting root node {}".format(node.identifier))
            adj = minimize(objective_fun(node.data.noised), node.data.attributes, 
                           constraints=cons, bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
            
            node.data.adjusted = adj.x
            node.data.error = node.data.attributes - node.data.adjusted

        # adjust children
        self.adjust_children(node, objective_fun, node_cons, bounds, parental_equality, maxiter, verbose)

        # recurse
        for child in self.children(node.identifier):
            self.__adjust_tree(child, objective_fun, node_cons, 
                                             bounds, parental_equality, maxiter, verbose)

    @dask.delayed
    def __adjust_tree_async(self, node, node_adj, objective_fun, node_cons, bounds, 
                          parental_equality, maxiter, verbose):
        """ Recursively noises children and then "adjusts" the children to sum
            up to the population of the parent.
        """
        
        if node.is_leaf():
            return {node.identifier: node_adj}

        # adjust children
        adj_children = self.__async_adjust_children(node, node_adj, objective_fun, node_cons, bounds,
                                                    parental_equality, maxiter, verbose)

        # recurse
        rs = dask.compute([self.__adjust_tree_async(self.get_node(child_id), adj_child, objective_fun, 
                                                    node_cons, bounds, parental_equality, maxiter, verbose) 
                           for child_id, adj_child in adj_children])[0]
        
        return {**{k: v for d in rs for k, v in d.items()}, **{node.identifier: node_adj}}

    def __async_adjust_children(self, node, adj_par, objective_fun, node_cons, bounds, 
                         parental_equality, maxiter, verbose):
        """ Adjusts the children to add up to the parent.
        """
#         time.sleep(random.random())
        children = self.children(node.identifier)
        num_attributes = node.data.attributes.shape[0]
        num_children = len(children)
        noised_children = np.append([], [child.data.noised for child in children])
        unnoised_children = np.append([], [child.data.attributes for child in children])
        
        bnds = [(0, None)]*(num_children*num_attributes) if bounds == "non-negative" else bounds(num_children)

        if parental_equality:
            cons_children = [{'type': 'eq', 'fun': lambda x: np.dot(adj_par - np.sum(x.reshape(num_children, num_attributes), axis=0),
                                                                    adj_par - np.sum(x.reshape(num_children, num_attributes), axis=0))}]
            if not node_cons:
                cons = cons_children
            else:
                cons = node_cons(num_children) + cons_children
        
        if verbose: print("Adjusting children of {}".format(node.identifier))
        adj = minimize(objective_fun(noised_children), unnoised_children, constraints=cons, 
                       bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
        adjusted_children = adj.x

        return [(children[i].identifier, adjusted_child) 
                for i, adjusted_child in enumerate(np.split(adjusted_children, num_children))]

    def __adjusted_root(self, root, objective_fun, node_cons, bounds, 
                        parental_equality, maxiter, verbose):
        # Returns adjusted root
        bnds = [(0, None)]*(root.data.attributes.shape[0]) if bounds == "non-negative" else bounds

        cons = node_cons if not node_cons else node_cons(1)
        if verbose: print("Adjusting root node {}".format(root.identifier))
        adj = minimize(objective_fun(root.data.noised), root.data.attributes, 
                       constraints=cons, bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
        
        return(adj.x)

    def __update_adjusted_and_error(self, adjusted_dict):
        for n in self.all_nodes_itr():
            n.data.adjusted = adjusted_dict[n.identifier]
            n.data.error = n.data.attributes - n.data.adjusted