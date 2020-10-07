from treelib import Node, Tree
from errors import SumError
import numpy as np
import math
import geopandas as gpd
# from gerrychain import Graph, Partition

class ToyDown(Tree):

    def __init__(self, filename, geoid_col, pop_col, parental_offsets=[3, 1, 6, 3, 2]):
        """
        """
        super(ToyDown, self).__init__()

        # self.graph = Graph.from_file(filename)
        self.gdf = gpd.read_file(filename)
        self.leaf_id = geoid_col
        self.pop_col = pop_col
        self.total_pop = self.gdf[self.pop_col].sum()
        self.levels = len(parental_offsets)

        geounits = self.create_tree_geounits_from_leaves(parental_offsets)
        self.populate_tree(geounits)
        self.add_levels_to_node(self.get_node(self.root), 0)
        self.print_unnoised_totaling_errors(self.get_node(self.root))

    def set_eps_budget_and_splits(self, eps_budget, eps_splits):
        """
        """
        self.eps_budget = eps_budget
        self.eps_values = self.epsilon_values(eps_splits, eps_budget)

    # def from_hierarchy(self, hierarchy, eps_budget, eps_splits):
    #     """ Initializes the Tree and populates it.
    #
    #         hierarchy  : List of tuples of the form (name, branching, population) sorted
    #                      in decreasing order of geographic nesting for eg.
    #                      [("Country", 1, 810), ("State", 3, 270), ("County", 3, 90), ("Dist", 3, 30)]
    #                      has 1 Country of population 810, 3 States of population 270 each,
    #                      3 Counties in each State of population 90 each and 3 Dists in each County of
    #                      population 30 each.
    #         eps_budget : Float, Epsilon budget across all levels
    #         eps_splits : List denoting the % of splits in epsilon value by level
    #                      eg. if the hierarchy is [Country, State, County, District] then
    #                      the eps_splits could look like [0, 0.33, 0.33, 0.34]
    #     """
    #     super(ToyDown, self).__init__()

        # self.eps_budget = eps_budget
        # self.eps_values = self.epsilon_values(eps_splits, eps_budget)

        # create nodes and populate the tree
        # geounits = []
        # self.add_geounits_at_level_to_list(hierarchy, 0, None, geounits)
        # self.populate_tree(geounits)
        # self.print_unnoised_totaling_errors(self.get_node(self.root))
        # self.add_levels_to_node(self.get_node(self.root), 0)

    def create_tree_geounits_from_leaves(self, parental_offsets):
        """
        """
        fliped_offsets = np.flip(parental_offsets)
        ls = np.cumsum(fliped_offsets)

        leafs = self.gdf[[self.leaf_id, self.pop_col]]
        nodes = self.gdf[[self.leaf_id, self.pop_col]]

        for i, offset in enumerate(np.cumsum(parental_offsets)[:-1]):
            level_names = leafs[self.leaf_id].apply(lambda s: s[:-offset])
            nodes = nodes.append(leafs.groupby(level_names).sum().reset_index(), ignore_index=True)

        node_dict = nodes.set_index(self.leaf_id).to_dict(orient="index")

        geounits = []
        for k ,v in node_dict.items():
            i, = np.where(ls == len(k))[0]
            par = k[:-fliped_offsets[i]]
            if par == "": par = None
            geounits.insert(0, self.GeoUnit(k, par, v[self.pop_col]))

        return geounits

    class GeoUnit(object):
        """ This class stores the data inside each Node in the tree.
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
            self.identifier = identifier if identifier else name

        def __repr__(self):
            name = self.__class__.__name__
            kwargs = [ "{}={}".format(k, v) for k, v in self.__dict__.items()]
            return "%s(%s)" % (name, ", ".join(kwargs))

    def add_geounits_at_level_to_list(self, hierarchy, level, parent, all_units):
        """ Recursively create the list of geounits needed to build the tree as specified in the `hierarchy`.

            Args:
                hierarchy : List of tuples of the form (name, branching, population) sorted
                            in decreasing order of geographic nesting for eg.
                            [("Country", 1, 810), ("State", 3, 270), ("County", 3, 90), ("Dist", 3, 30)]
                            has 1 Country of population 810, 3 States of population 270 each,
                            3 Counties in each State of population 90 each and 3 Dists in each County of
                            population 30 each.
                level     : Int, index of the `hierarchy` we are currently at
                parent    : Name of the parent of the GeoUnits we are building
                all_units : List of GeoUnits that stores all the GeoUnits built in this recursive function.

        """
        if level == len(hierarchy):
            return
        if level == 0:
            # build root
            name, branching, pop = hierarchy[level]
            all_units.append(self.GeoUnit(name, None, pop))
            self.add_geounits_at_level_to_list(hierarchy, level+1, name, all_units)
        else:
            name, branching, pop = hierarchy[level]
            for i in range(branching):
                curr_name = parent + name + str(i+1)
                all_units.append(self.GeoUnit(curr_name, parent, pop))
                self.add_geounits_at_level_to_list(hierarchy, level+1, curr_name, all_units)

    def epsilon_values(self, eps_splits, eps_budget):
        """ Stores the epsilon values as a List of Floats.
            The eps_values list should be in decreasing order of size
            eg [Country, State, County, Tract, BlockGroup, Block]
        """
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
        assert(epsilon > 0)

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

            Also add the "error" attribute to the node, which is the difference
            between the adjusted population and the unnoised population of the node.
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
            child.data.error = child.data.adjusted_pop - child.data.unnoised_pop

    def noise_and_adjust(self):
        """ Noises each node in the Tree and adjusts them to add back to their parent.
            This function simply serves as a wrapper function to the recursive
            __noise_and_adjust_children(), and is started at the root of the tree.
        """
        root = self.get_node(self.root)
        self.__noise_and_adjust_children(root)
        self.print_adjusted_totaling_errors(root)

    def __noise_and_adjust_children(self, node):
        """ Recursively noises children and then "adjusts" the children to sum
            up to the population of the parent.

            If at root, noises + adjusts both the root and the children of the root.
        """
        if node.is_leaf():
            return
        elif node.is_root():
            # add noise to root. No adjustment is done on the root.
            self.add_laplacian_noise(node, self.eps_values[node.data.level])
            node.data.adjusted_pop = node.data.noised_pop
            node.data.error = node.data.adjusted_pop - node.data.unnoised_pop

        # noise and adjust
        self.noise_children(node)
        self.adjust_children(node)

        # recurse
        for child in self.children(node.identifier):
            self.__noise_and_adjust_children(child)

    def print_adjusted_totaling_errors(self, node, abs_tol=0.00005):
        """
            Prints the node's name if the ".adjusted_pop" attribute of the
            node's children does not sum up to it's own ".adjusted_pop" upto
            a precision of `abs_tol`.
        """
        if node.is_leaf():
            return
        else:
            children_total = sum([child.data.adjusted_pop for child in self.children(node.identifier)])

            if not math.isclose(node.data.adjusted_pop, children_total, abs_tol=abs_tol):
                raise SumError("Expected {} but the children totaled " \
                               "to {} for node {}".format(node.data.adjusted_pop,
                                                          children_total,
                                                          node.tag))

            for child in self.children(node.identifier):
                self.print_adjusted_totaling_errors(child)

    def print_unnoised_totaling_errors(self, node, abs_tol=0.00005):
        """
            Prints the node's name if the ".unnoised_pop" attribute of the
            node's children does not sum up to it's own ".unnoised_pop" upto
            a precision of `abs_tol`.
        """
        if node.is_leaf():
            return
        else:
            children_total = sum([child.data.unnoised_pop for child in self.children(node.identifier)])

            if not math.isclose(node.data.unnoised_pop, children_total, abs_tol=abs_tol):
                raise SumError("Expected {} but the children totaled " \
                               "to {} for node {}".format(node.data.unnoised_pop,
                                                          children_total,
                                                          node.tag))

            for child in self.children(node.identifier):
                self.print_unnoised_totaling_errors(child)

    def assign_district_tree_variance(self, district, eps=None, eps_splits=None, sensitivity=2):
        """
        """
        self.assign_district_to_leaves(district)
        root = self.get_node(self.root)
        print(root)
        self.assign_weights(root)

        epsilons = np.ones(self.levels)*np.sqrt(2) if eps_splits == None else np.array(eps_splits)
        if eps != None:
            epsilons *= eps if eps_splits else eps*(1/np.sqrt(2))*(1/self.levels)

        return self.district_variance(root, epsilons, sensitivity)

    def assign_district_to_leaves(self, district):
        """
        """
        for leaf, weight in district.items():
            self.get_node(leaf).data.weight = weight

    def assign_weights(self, node):
        """
        """
        if not node.is_leaf():
            children = self.children(node.identifier)
            child_weights = [self.assign_weights(child) for child in children]
            node.data.weight = np.mean(child_weights)
        return node.data.weight

    def district_variance(self, node, epsilons, sensitivity):
        """
        """
        eps_k = epsilons[0]

        if node.is_leaf():
            child_vars = []
        else:
            children = self.children(node.identifier)
            child_vars = [self.district_variance(child, epsilons[1:], sensitivity) for child in children]

        if node.data.parent == None:
            return node.data.weight**2 * 2*(sensitivity/eps_k)**2 + sum(child_vars)
        else:
            par_weight = self.get_node(node.data.parent).data.weight
            return (node.data.weight - par_weight)**2 * 2*(sensitivity/eps_k)**2 + sum(child_vars)
