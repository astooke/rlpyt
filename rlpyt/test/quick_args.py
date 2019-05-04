

class BaseTestCls(object):

    def __init__(self, value1, value2, value3=3, value4=dict(x=1)):
        breakpoint()


class InheritTestCls(BaseTestCls):

    def __init__(self, value10, value1, *args, **kwargs):
        breakpoint()
        super().__init__(*args, **kwargs)
        breakpoint()


if __name__ == "__main__":
    a = InheritTestCls(0, "a", 4)
