""" 1940 constraint creator class """

import math
import itertools
import numpy as np
import programs.engine.cenquery as cenquery

class AbstractConstraintsCreator:
    """ New super class for constraint creators
        This class is what is used to create constraint objects from what is listed in the config file
    Inputs:
        hist_shape: the shape of the underlying histogram
        invariants: the list of invariants (already created)
        constraint_names:  the names of the constraints to be created (from the list below)
    """
    def __init__(self, hist_shape, invariants, constraint_names):
        self.invariants = invariants
        self.constraint_names = constraint_names
        self.hist_shape = hist_shape
        self.constraints_dict = {}
        self.constraint_funcs_dict = {}

    def calculateConstraints(self):
        for name in self.constraint_names:
            assert name in self.constraint_names, "Provided constraint name '{}' not found.".format(name)
            self.constraint_funcs_dict[name]()
        return self



class ConstraintsCreator1940(AbstractConstraintsCreator):
    """
    This class is what is used to create constraint objects from what is listed in the config file
    Inputs:
        hist_shape: the shape of the underlying histogram
        invariants: the list of invariants (already created)
        constraint_names: the names of the constraints to be created (from the list below)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.constraint_funcs_dict = {
            "total"                     : self.total,
        }

       
    
    # Total population per geounit must remain invariant
    def total(self):
        subset = None
        add_over_margins = (0, 1, 2, 3)
        query = cenquery.Query(array_dims=self.hist_shape,subset=subset, add_over_margins=add_over_margins)
        rhs = self.invariants["tot"].astype(int)
        sign = "="
        self.constraints_dict["total"] = cenquery.Constraint(query=query, rhs=rhs, sign=sign, name="total")


