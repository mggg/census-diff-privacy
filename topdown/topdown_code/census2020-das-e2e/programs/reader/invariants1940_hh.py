""" Module to store swappable invariant creator classes """

import numpy as np
import programs.engine.cenquery as cenquery

class AbstractInvariantsCreator:
    """ New super class for invariant creators """
    def __init__(self, raw, raw_housing, invariant_names):
        self.raw = raw.toDense()
        self.raw_housing = raw_housing.toDense()
        self.invariant_names = invariant_names
        self.invariant_funcs_dict = {}
        self.invariants_dict = {}

    def calculateInvariants(self):
        for name in self.invariant_names:
            assert name in self.invariant_funcs_dict, "Provided invariant name '{}' not found.".format(name)
            self.invariant_funcs_dict[name]()
        return self

class InvariantsCreator1940(AbstractInvariantsCreator):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.invariant_funcs_dict = {
            "tot"                       : self.tot,
        }


    def tot(self):
        data = self.raw
        add_over_margins = (0, 1, 2, 3)
        subset = None
        query = cenquery.Query(array_dims=data.shape, subset=subset, add_over_margins=add_over_margins)
        self.invariants_dict["tot"] = np.array(query.answer(data)).astype(int)

