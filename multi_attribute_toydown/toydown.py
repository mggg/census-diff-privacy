import numpy as np
import scipy as sp
from scipy.optimize import minimize
import treelib
from treelib import Node, Tree
import gurobipy as gp
from gurobipy import GRB

class GeoUnit(object):
    """ This class stores the data inside each Node in the tree.
    """
    def __init__(self, name, parent, attributes, identifier=None):
        """ Args:
                name         : Str, Name of the Node
                parent       : Str, The identifier of the parent of the Node
                attributes   : if using scipy; np.array(dtype=int), The array of population counts for
                               the Node.
                               if using gurobi: dictionary of count names : np.array(dtype=int).  where 
                               the first entry is the sum of the following entries.
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
    def __init__(self, geounits, num_levels, eps_budget, eps_splits, gurobi=False, pop_vap=True):
        """ Initializes the Tree and populates it.
            geounits   : List of GeoUnits that will form the nodes of the Tree.
            num_levels : The height of the tree
            eps_budget : Float, Epsilon budget across all levels
            eps_splits : List denoting the % of splits in epsilon value by level
                         eg. if the hierarchy is [Country, State, County, District] then
                         the eps_splits could look like [0, 0.33, 0.33, 0.34]
            gurobi     : If True use gurobi solver, else use scipy solver
        """
        super(ToyDown, self).__init__()
        self.eps_budget = eps_budget
        self.populate_tree(geounits)
        self.add_levels_to_node(self.get_node(self.root), 0)
        self.eps_values = self.epsilon_values(num_levels, eps_splits, eps_budget)
        self.gurobi = gurobi
        self.pop_vap = pop_vap

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


    def add_laplacian_noise(self, node, epsilon):
        """ Adds Laplacian noise of parameter 1/`epsilon` to Node `node`. If `epsilon` is np.inf, 
            adds no noise.  `epsilon` must be positive, otherwise throws ValueError
        """
        if epsilon <= 0: raise ValueError("epsilon <= 0")

        if self.gurobi:
             node.data.noise = {}
             node.data.noised = {}
             
             for k, v in node.data.attributes.items():
                shape = v.shape
                noise = np.random.laplace(scale=1/epsilon, size=shape)
                node.data.noise[k] = noise
                node.data.noised[k] = v + noise

        else:
            shape = node.data.attributes.shape
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

    def noise_and_adjust(self, objective_fun="L1", node_cons=None, bounds="non-negative", 
                         parental_equality=True, maxiter=200, verbose=False, save=False):
        """ Noises each node in the Tree and adjusts them to add back to their parent.
            This function simply serves as a wrapper function to the recursive 
            __noise_and_adjust_children(), and is started at the root of the tree.

            objective_fun     : Function to minimize over.  If "L1" uses the L1 norm, 
                                if "L2" uses the L2 norm, otherwise a function that takes a
                                noised point, and returns a function that takes a point x and returns 
                                a scalar distance between the two points.
            node_cons         : Function that takes the number of children, n, and returns a list of constraints 
                                for the readjustment of a Node is subject to. 
                                Has form [{"type": "eq/ineq", "fun": lambda x: }]
                                See scipy.optimize constraint specification for more details.
            bounds            : If using scipy.optimzize - Function that takes the number of children, n, and returns a list of 
                                (min, max) pairs for each attribute in each of the children, defining 
                                the bounds on that attribute. Use None for one of min or max when there is 
                                no bound in that direction.
                                A value of "non-negative", flags that all counts should be > 0
                                If using gurobi, a value of None, imposes no bounds on the count values.
                                and any other value, flags that all counts should be > 0
            parental_equality : Boolean flag - adds constraint that for all attributes, the sum of the 
                                children's counts should be equal to that of the parent's counts.
            maxiter           : Option to pass along to scipy optimizer.  Defines the maximum number of
                                iterations the optimizer should preform.  Default value 200.
            verbose           : Boolean flag -- if True prints debug info.
        """

        root = self.get_node(self.root)
        
        # noise tree
        self.noise_tree(root)

        if self.gurobi:
            adj_root = self.adjusted_root_gp(root, None, None, bounds, verbose, self.pop_vap)
        else:
            if objective_fun == "L1": objective_fun = lambda n: lambda x: sp.linalg.norm(x-n, ord=1)
            if objective_fun == "L2": objective_fun = lambda n: lambda x: sp.linalg.norm(x-n, ord=2)
            adj_root = self.adjusted_root(root, objective_fun, node_cons, bounds,
                                      maxiter, verbose)

        adjusted = self.__adjust_tree(self, self.root, adj_root, objective_fun, node_cons, bounds,
                           parental_equality, maxiter, verbose, self.gurobi, self.pop_vap)

        if save: self.update_adjusted_and_error(adjusted)
        return adjusted

    def noise_tree(self, node):
        self.add_laplacian_noise(node, self.eps_values[node.data.level])

        if node.is_leaf():
            return
        
        for child in self.children(node.identifier):
            self.noise_tree(child)

    @staticmethod
    def __adjust_tree(model, node_id, node_adj, objective_fun, node_cons, bounds, 
                                    parental_equality, maxiter, verbose, gurobi, pop_vap):
        """ Recursively noises children and then "adjusts" the children to sum
            up to the population of the parent.
        """
        if model.get_node(node_id).is_leaf():
            return {node_id: node_adj}

        # adjust children
        if gurobi:
             adj_children = ToyDown.adjust_children_gp(model.children(node_id), node_adj, bounds,
                                                            parental_equality, verbose, pop_vap)
        else:
            adj_children = ToyDown.adjust_children(model.children(node_id), node_adj, objective_fun, 
                                               node_cons, bounds, parental_equality, maxiter, verbose)


        # recurse
        attribute_dict = {node_id: node_adj}
        
        for child_id, adj_child in adj_children:
            adjusted_child =  ToyDown.__adjust_tree(treelib.Tree(model).subtree(child_id), 
                                                    child_id, adj_child, objective_fun, 
                                                    node_cons, bounds, parental_equality, 
                                                    maxiter, verbose, gurobi, pop_vap)
            attribute_dict.update(adjusted_child)
    
        return attribute_dict


    @staticmethod
    def adjust_children(children, adj_par, objective_fun, node_cons, bounds, 
                        parental_equality, maxiter, verbose):

        num_attributes = adj_par.shape[0]
        num_children = len(children)
        noised_children = np.append([], [child.data.noised for child in children])
        unnoised_children = np.append([], [child.data.attributes for child in children])

        bnds = [(0, None)]*(num_children*num_attributes) if bounds == "non-negative" else bounds(num_children)

        if parental_equality:
            cons_children = [{'type': 'eq', 'fun': lambda x: np.dot(adj_par - np.sum(x.reshape(num_children, 
                                                                                     num_attributes), axis=0),
                                                                    adj_par - np.sum(x.reshape(num_children, 
                                                                                     num_attributes), axis=0))}]
            if not node_cons:
                cons = cons_children
            else:
                cons = node_cons(num_children) + cons_children

        adj = minimize(objective_fun(noised_children), unnoised_children, constraints=cons, 
                       bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
        adjusted_children = adj.x

        return [(children[i].identifier, adjusted_child) 
            for i, adjusted_child in enumerate(np.split(adjusted_children, num_children))]


    def adjusted_root(self, root, objective_fun, node_cons, bounds, 
                      maxiter, verbose):
        # Returns adjusted root
        bnds = [(0, None)]*(root.data.attributes.shape[0]) if bounds == "non-negative" else bounds

        cons = node_cons if not node_cons else node_cons(1)
        if verbose: print("Adjusting root node {}".format(root.identifier))
        adj = minimize(objective_fun(root.data.noised), root.data.attributes, 
                       constraints=cons, bounds=bnds, options={"maxiter": maxiter, "disp": verbose})
        
        return(adj.x)


    @staticmethod
    def adjusted_root_gp(root, objective_fun, node_cons, bounds, verbose, pop_vap):
        # Returns adjusted root
        m = gp.Model("root")
        if not verbose: m.params.OutputFlag = 0

        ks = []
        vs = []
        pop_indexs = [0,0] ## POP index, VAP index
        for i, (k, v) in enumerate(root.data.noised.items()):
            ks.append(k)
            vs.append(v)
            if k == "TOTPOP": pop_indexs[0] = i
            if k == "VAP": pop_indexs[1] = i

        root_noised = np.array(vs)
        num_cols = root_noised.shape[1]
        num_rows = root_noised.shape[0]

        if bounds:
            x = m.addVars(num_rows, num_cols, vtype=GRB.CONTINUOUS, name="Counts")
        else:
            x = m.addVars(num_rows, num_cols, lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                          vtype=GRB.CONTINUOUS, name="Counts")
        
        exps = []
        for i in range(num_rows):
            for j in range(num_cols):
                exps.append((x[i,j] - root_noised[i,j])*(x[i,j] - root_noised[i,j]))

        obj = gp.quicksum(exps)
        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs((x[i,0] == gp.quicksum([x[i,j] for j in range(1, num_cols)]) for i in range(num_rows)), 
                         name='node_cons')

        if pop_vap:
            m.addConstrs((x[pop_indexs[0], j] >= x[pop_indexs[1], j] for j in range(1, num_cols)), 
                         name='vap_leq_pop')

        m.optimize()
        adj_root_atts = np.array([v.x for v in m.getVars()]).reshape(root_noised.shape)
        
        return {ks[i]: adj_root_atts[i] for i in range(num_rows)}

    @staticmethod
    def adjust_children_gp(children, adj_par, bounds, parental_equality, verbose, pop_vap):
        """By default uses the least squares (L2 norm) as the objective function.  And node_cons
           are that node_atributes[0] == sum(node_atributes[1:]) """
        
        ks = []
        par_vs = []
        pop_indexs = [0,0] ## POP index, VAP index
        for i, (k, v) in enumerate(adj_par.items()):
            ks.append(k)
            par_vs.append(v)
            if k == "TOTPOP": pop_indexs[0] = i
            if k == "VAP": pop_indexs[1] = i

        par_adjusted = np.array(par_vs)
        num_rows = par_adjusted.shape[0]
        num_cols = par_adjusted.shape[1]
        

        # num_cols = adj_par.shape[0]
        num_children = len(children)
        children_noised = np.zeros((num_children, num_rows, num_cols))
        for c in range(num_children):
            for i in range(num_rows):
                children_noised[c,i] = children[c].data.noised[ks[i]]
        # children_noised = np.array([child.data.noised for child in children])

        m = gp.Model("children")
        if not verbose: m.params.OutputFlag = 0
        if bounds:
            xs = m.addVars(num_children, num_rows, num_cols,vtype=GRB.CONTINUOUS, name="Counts")
        else:
            xs = m.addVars(num_children, num_rows, num_cols, lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                           vtype=GRB.CONTINUOUS, name="Counts")
        
        exps = []
        for c in range(num_children):
            for i in range(num_rows):
                for j in range(num_cols):
                    exps.append((xs[c,i,j] - children_noised[c,i,j])*(xs[c,i,j] - children_noised[c,i,j]))

        obj = gp.quicksum(exps)
        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs((xs[c,i,0] == gp.quicksum([xs[c,i,j] for j in range(1, num_cols)]) for i in range(num_rows)
                                                                                        for c in range(num_children)), 
                     name='node_cons')

        if parental_equality:
            m.addConstrs((par_adjusted[i,j] == gp.quicksum([xs[c,i,j] for c in range(num_children)]) for j in range(num_cols)
                                                                                                     for i in range(num_rows)), 
                name="parental_equality")
        
        if pop_vap:
            m.addConstrs((xs[c, pop_indexs[0], j] >= xs[c, pop_indexs[1], j] for j in range(num_cols) for c in range(num_children)), 
                         name='vap_leq_pop')

        m.optimize()
        adj_children = np.array([v.x for v in m.getVars()]).reshape(num_children, num_rows, num_cols)

        return [(children[i].identifier, {ks[j]: adjusted_child[j] for j in range(num_rows)}) 
                for i, adjusted_child in enumerate(adj_children)]


    def update_adjusted_and_error(self, adjusted_dict):
        for n in self.all_nodes_itr():
            n.data.adjusted = adjusted_dict[n.identifier]
            n.data.error = n.data.attributes - n.data.adjusted
