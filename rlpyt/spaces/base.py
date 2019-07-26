

class Space():
    """
    Common definitions for observations and actions.
    """

    def sample(self):
        """
        Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError

    def null_value(self):
        """
        Return a null value used to fill for absence of element.
        """
        raise NotImplementedError
