from typing import Type, Set, Dict, List, Any
from bidict import bidict


class Namer:
    def __init__(self):
        self._type_short_names: bidict[Type, str] = bidict()
        self._obj_names: Dict[Type, bidict[Any, str]] = dict()
        self.separator = "_"
        self.__initialize()

    def __initialize(self):
        from Aipiler import FakeTensor, FakeScalar, Parameter
        from Aipiler.dim import Dim
        from Aipiler.axis import Axis
        from Aipiler.primitive import MapPrimitive, ReducePrimitive, UnaryPrimitive

        for ty, ty_name in (
            (FakeTensor, "tensor"),
            (FakeScalar, "scalar"),
            (Parameter, "param"),
            (Dim, "dim"),
            (Axis, "axis"),
            (MapPrimitive, "map"),
            (ReducePrimitive, "reduce"),
            (UnaryPrimitive, "unary"),
        ):
            self._obj_names.update({ty: bidict()})
            self._type_short_names.update({ty: ty_name})

    def register_type(self, ty: Type, short_name: str):
        if ty in self._type_short_names and self._type_short_names[ty] != short_name:
            raise ValueError(
                f"{ty} is already registed with name: {self._type_short_names[ty]}"
            )
        if (
            short_name in self._type_short_names.inverse
            and self._type_short_names.inverse[short_name] != ty
        ):
            raise ValueError(
                f"{short_name} is already registed by type: {self._type_short_names.inverse[short_name]}"
            )
        if self.separator in short_name:
            raise ValueError(
                f"Short Name: {short_name} has separator: {self.separator}"
            )
        if ty in self._obj_names:
            return
        else:
            self._obj_names[ty] = bidict()
            self._type_short_names[ty] = short_name

    def get_or_create_name_of(self, obj: Any):
        ty = type(obj)
        # if type(obj) is not registered, raise error
        if ty not in self._obj_names:
            raise ValueError(f"Unregistered type: {ty}")
        _obj_names = self._obj_names[ty]
        # if obj is not registered, create new name and record it
        if obj not in _obj_names:
            ty_name = self._type_short_names[ty]
            obj_name = f"{ty_name}{self.separator}{len(_obj_names)}"
            _obj_names[obj] = obj_name
            return obj_name
        return _obj_names[obj]

    def lookup_name(self, obj):
        ty = type(obj)
        # if type(obj) is not registered, raise error
        if ty not in self._obj_names:
            raise ValueError(f"Unregistered type: {ty}")
        _obj_names = self._obj_names[ty]
        # if obj is not registered, create new name and record it
        if obj not in _obj_names:
            raise ValueError(f"Cannot find name of {obj}")
        return _obj_names[obj]

    def lookup_obj(self, name: str):
        ty: Type = None
        ty_name = name.split(self.separator)[0]
        for registed_name in self._type_short_names.values():
            if ty_name in registed_name:
                ty = self._type_short_names.inverse[registed_name]
        if ty is None:
            raise ValueError("Cannot find type with name: ")
        _objs = self._obj_names[ty]
        if name not in _objs.inverse:
            raise ValueError("Cannot find {name}: {ty_name}")
        return _objs.inverse[name]


N = Namer()

if __name__ == "__main__":
    N.register_type(str, "s")
    obj1 = "abc"
    n1 = N.get_or_create_name_of(obj1)
    obj2 = "bcd"
    n2 = N.get_or_create_name_of(obj2)
    print(n1, n2)
    print(obj1 is N.lookup_obj(n1))
