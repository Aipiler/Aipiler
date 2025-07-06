class Axis:
    def __init__(
        self,
        name: str,
        from_prim,
        is_from_input: bool,
        idx_in_script: int,
        idx_of_script: int,
    ):
        from Aipiler.primitive import EinsumPrimitive

        self._name = name
        self._from_prim: EinsumPrimitive = from_prim
        self._is_from_input = is_from_input
        self._idx_in_script = idx_in_script
        self._idx_of_script = idx_of_script

    @property
    def name(self):
        return self._name

    @property
    def from_input(self):
        return self._is_from_input

    @property
    def index_of_script(self):
        return self._idx_of_script

    @property
    def index_in_script(self):
        return self._idx_in_script

    @property
    def is_combined(self):
        return len(self._name) > 1

    @property
    def is_scalar(self):
        return self._name == "_"

    def __repr__(self):
        from Aipiler.utils.namer import N

        return "{}({}-th of {}{} in {})".format(
            self._name,
            self._idx_in_script,
            "In" if self._is_from_input else "Out",
            self._idx_of_script,
            N.get_or_create_name_of(self._from_prim),
        )

    def eq_with(self, another: "Axis"):
        return self._name == another._name

    def __contains__(self, lhs: "Axis"):
        if not isinstance(lhs, Axis):
            raise ValueError("Axis object expected ")
        return lhs._name in self._name
