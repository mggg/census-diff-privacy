from treelib import Node, Tree
import numpy as np

class GeoUnit(object):
    """ This class stores the data required inside each Node in the tree.
    """
    def __init__(self, name, parent, unnoised_pop, identifier=None):
        """ Args:
                name         : Str, Name of the Node
                parent       : Str, The identifier of the parent of the Node
                unnoised_pop : Int, Unnoised population
                identifier   : Str, A string that is unique to the Node in the entire Tree
        """
        self.name = name
        self.unnoised_pop = unnoised_pop
        self.parent = parent
        if identifier:
            self.identifier = identifier
        else:
            self.identifier = name

class ToyClown(Tree):
    def __init__(self, geounits, num_levels, eps_budget, eps_splits):
        """ Initializes the Tree and populates it.

            geounits   : List of GeoUnits that will form the nodes of the Tree.
            eps_budget : Float, Epsilon budget across all levels
            eps_splits : List denoting the % of splits in epsilon value by level
                         eg. if the hierarchy is [Country, State, County, District] then
                         the eps_splits could look like [0, 0.33, 0.33, 0.34]
        """
        super(ToyClown, self).__init__()
        self.eps_budget = eps_budget
        self.populate_tree(geounits)
        self.add_levels_to_node(self.get_node(self.root), 0)
        self.eps_values = self.epsilon_values(num_levels, eps_splits, eps_budget)

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
        """ Adds Laplacian noise of parameter 1/`epsilon` to Node `node`. If `epsilon` is 0, 
            adds no noise.
        """
        if epsilon == 0:
            noise = 0
        else:
            noise = np.random.laplace(loc=0, scale=1/epsilon)

        node.data.noised_pop = node.data.unnoised_pop + noise
        node.data.noise = noise
        node.data.noise_type = "laplacian"

    def add_levels_to_node(self, node, level):
        """ Recursive function that adds the `level` as an attribute
            to the `node`.
        """
        self.get_node(node.identifier).data.level = level
        for child in self.children(node.identifier):
            self.add_levels_to_node(child, level+1)

    def noise_children(self, node):
        """ Adds noise to each child of `node`.
        """
        for child in self.children(node.identifier):
            self.add_laplacian_noise(child, self.eps_values[child.data.level])

    def adjust_children(self, node):
        """ Adjusts the children to add up to the parent.
        """
        noised = 0

        for child in self.children(node.identifier):
            noised += child.data.noised_pop

        # calculate the constant that gets added/subtracted from each node. This is denoted as
        # (Q-P)/n in Proposition 1 of the working paper.
        constant = (node.data.adjusted_pop - noised) / len(self.children(node.identifier))

        # go to each child, add/subtract that constant
        for child in self.children(node.identifier):
            child.data.adjusted_pop = child.data.noised_pop + constant

    def noise_and_adjust(self):
        """ Noises each node in the Tree and adjusts them to add back to their parent.
            This function simply serves as a wrapper function to the recursive 
            __noise_and_adjust_children(), and is started at the root of the tree.
        """
        self.__noise_and_adjust_children(self.get_node(self.root))

    def __noise_and_adjust_children(self, node):
        """ Recursively noises children and then "adjusts" the children to sum
            up to the population of the parent.
        """
        if node.is_leaf():
            return
        elif node.is_root():
            # add noise to root. No adjustment is done on the root.
            self.add_laplacian_noise(node, self.eps_values[node.data.level])
            node.data.adjusted_pop = node.data.noised_pop

        # noise and adjust
        self.noise_children(node)
        self.adjust_children(node)

        # recurse
        for child in self.children(node.identifier):
            self.__noise_and_adjust_children(child)
