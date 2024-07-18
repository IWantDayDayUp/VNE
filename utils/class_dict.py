import copy


class ClassDict(object):

    def __init__(self) -> None:
        super(ClassDict, self).__init__()

    def update(self, *args, **kwargs):
        """Update the ClassDict object with key-value pairs from a dictionary or another ClassDict object"""
        for k, v in kwargs.items():
            self[k] = v
            
    @classmethod
    def from_dict(cls, dict):
        """Create a new ClassDict object from a dictionary"""
        cls.__init__ = dict
        return cls
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int):
            print(key)
            return super().__getitem__(key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)