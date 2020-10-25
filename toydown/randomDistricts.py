import numpy as np
import scipy as sp
import math
from scipy.optimize import minimize
import treelib
from treelib import Node, Tree
from gerrychain import Graph
import geopandas as gpd
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
# import dask
# import gurobipy as gp
# from gurobipy import GRB

class Hierarchy_2D:
    def __init__(self, filename, geoid_col, pop_col, parental_offsets=[3, 1, 6, 3, 2]):
        """ filename - path to shapefile .shp
            geoid_col - name of column
        """
        # self.graph = Graph.from_file(filename)
        self.gdf = gpd.read_file(filename)
        self.leaf_id = geoid_col
        self.pop_col = pop_col
        self.total_pop = self.gdf[self.pop_col].sum()
        self.levels = len(parental_offsets)

        self.tree = self.Hierarchy_Tree(self.create_tree_geounits_from_leaves(parental_offsets))


    def create_tree_geounits_from_leaves(self, parental_offsets):
        """
        """
        fliped_offsets = np.flip(parental_offsets)
        ls = np.cumsum(fliped_offsets)

        leafs = self.gdf[[self.leaf_id, self.pop_col]]
        nodes = self.gdf[[self.leaf_id, self.pop_col]]

        for i, offset in enumerate(np.cumsum(parental_offsets)[:-1]):
            level_names = leafs[self.leaf_id].apply(lambda s: s[:-offset])
            nodes = nodes.append(leafs.groupby(level_names).sum().reset_index(),
                                 ignore_index=True)

        node_dict = nodes.set_index(self.leaf_id).to_dict(orient="index")

        geounits = []
        for k ,v in node_dict.items():
            i, = np.where(ls == len(k))[0]
            par = k[:-fliped_offsets[i]]
            if par == "": par = None
            geounits.insert(0, self.GeoUnit(k, par, v))

        return geounits

    def build_district(self, num_districts, epsilon=0.02, updaters=None):
        """
            build_district - generates a random district and returns a Partition object
            and a dictionary mapping geoids to district assignments representing that district.
            The district 1, is the created district.

            Parameters:
                num_dictricts - int k, such if the district were in a plan there would be k
                                districts.  aka. the district build has 1/k of the total population
                epsilon       - float between 0 and 1, how much deviance from its ideal population
                                the district is allowed to have.
                                Default value: 2%
                updaters      - dict, updaters to pass along to the partition object.  If None
                                sets updaters "cut_edges" and "population", a Tally of the pop_col
        """
        pop_targets = [self.total_pop/num_districts]
        d  = self.grow_districts(self.graph, self.pop_col, pop_targets, self.total_pop, epsilon)
        # print(pop_targets, d)

        ps = []
        for i in d:
            if updaters:
                ps.append(Partition(self.graph, {x:int(x in d[i]) for x in self.graph.nodes},
                                updaters=updaters))
            else:
                ps.append(Partition(self.graph, {x:int(x in d[i]) for x in self.graph.nodes},
                                updaters={"cut_edges": cut_edges, "population": Tally(self.pop_col)}))

        part = ps[0]
        mapping = {self.graph.nodes[x][self.leaf_id]:int(x in d[0]) for x in self.graph.nodes}

        return (part, mapping)

    def assign_district_tree_variance(self, district, eps=None, eps_splits=None, sensitivity=2):
        self.tree.assign_district_to_leaves(district)
        root = self.tree.get_node(self.tree.root)
        self.tree.assign_weights(root)

        epsilons = np.ones(self.levels)*np.sqrt(2) if eps_splits == None else np.array(eps_splits)
        if eps != None:
            epsilons *= eps if eps_splits else eps*(1/np.sqrt(2))*(1/self.levels)

        return self.tree.district_variance(root, epsilons, sensitivity)

    @staticmethod
    def grow_districts(graph, pop_col, pop_targets, total_pop, epsilon):
        '''
        Builds districts corresponding to pop_targets.

        pop_targets MUST BE SORTED!
        '''
        pop_targets.append(math.inf)
        districts = {}
        target_index = 0
        popshare = [graph.nodes[x][pop_col]/total_pop for x in graph.nodes]
        start = np.random.choice(range(len(popshare)), p=popshare)
        district_list = [start]
        frontier = set([start])
        pop = graph.nodes[start][pop_col]

        while target_index < len(pop_targets)-1:
            add_node = np.random.choice(list(frontier))
            pop += sum([graph.nodes[n][pop_col] for n in set(graph.neighbors(add_node))-set(district_list)])
            frontier = frontier.union(set(graph.neighbors(add_node))-set(district_list))
            frontier = frontier - set([add_node])
            district_list.extend(set(graph.neighbors(add_node)) - set(district_list))
            while pop_targets[target_index]*(1-epsilon) < pop:
                #over current population target
                if pop_targets[target_index]*(1+epsilon) > pop:
                    districts[target_index] = district_list.copy()
                target_index += 1
        pop_targets.pop()
        return districts

    class GeoUnit(object):
        """ This class stores the data inside each Node in the tree.
        """
        def __init__(self, name, parent, attributes=None, identifier=None):
            """ Args:
                    name         : Str, Name of the Node
                    parent       : Str, The identifier of the parent of the Node
                    attributes   : dict of node counts
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


    class Hierarchy_Tree(Tree):
        def __init__(self, geounits):
            """ Initializes the Tree and populates it.
                geounits   : List of GeoUnits that will form the nodes of the Tree.
            """
            super(type(self), self).__init__()
            self.populate_tree(geounits)
            self.add_levels_to_node(self.get_node(self.root), 0)


        def populate_tree(self, geounits):
            """ Populates the Tree from a list of GeoUnits.
            """
            for unit in geounits:
                if unit.parent:
                    self.create_node(unit.name, unit.identifier, parent=unit.parent, data=unit)
                else:
                    # root node
                    self.create_node(unit.name, unit.identifier, data=unit)


        def add_levels_to_node(self, node, level):
            """ Recursive function that adds the `level` as an attribute
                to the `node`.
            """
            self.get_node(node.identifier).data.level = level
            for child in self.children(node.identifier):
                self.add_levels_to_node(child, level+1)


        def assign_district_to_leaves(self, district):
            for leaf, weight in district.items():
                self.get_node(leaf).data.weight = weight

        def assign_weights(self, node):
            if not node.is_leaf():
                children = self.children(node.identifier)
                child_weights = [self.assign_weights(child) for child in children]
                node.data.weight = np.mean(child_weights)
            return node.data.weight


        def district_variance(self, node, epsilons, sensitivity):
            eps_k = epsilons[0]
            if node.is_leaf():
                child_vars = []
            else:
                children = self.children(node.identifier)
                child_vars = [self.district_variance(child, epsilons[1:]) for child in children]

            if node.data.parent == None:
                return node.data.weight**2 * 2*(sensitivity/eps_k)**2 + sum(child_vars)
            else:
                par_weight = self.get_node(node.data.parent).data.weight
                return (node.data.weight - par_weight)**2 * 2*(sensitivity/eps_k)**2 + sum(child_vars)
