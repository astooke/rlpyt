

class Space():
    """
    Common definitions for observations and actions.
    """

    def sample(self, n=None):
        """
        Uniformly randomly sample a random element(s) of this space
        """
        raise NotImplementedError
