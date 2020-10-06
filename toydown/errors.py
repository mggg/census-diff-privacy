
class SumError(Exception):
    """ Raised when an attribute of the children of a node don't add up to
        the node's attribute.
    """

    def __init__(self, message):
        """
        """
        self.message = message
