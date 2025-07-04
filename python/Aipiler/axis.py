class Axis:
    def __init__(
        self, name: str, from_input: bool, idx_in_script: int, idx_of_script: int
    ):
        self._name = name
        self._from_input = from_input
        self._idx_in_script = idx_in_script
        self._idx_of_script = idx_of_script

    @property
    def name(self):
        return self._name

    @property
    def from_input(self):
        return self._from_input

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
        return (
            f"I{self._idx_of_script}_{self._name}{self._idx_in_script}"
            if self._from_input
            else f"O{self._idx_of_script}_{self._name}{self._idx_in_script}"
        )

    def eq_with(self, another: "Axis"):
        return self._name == another._name

    def __contains__(self, lhs: "Axis"):
        if not isinstance(lhs, Axis):
            raise ValueError("Axis object expected ")
        return lhs._name in self._name
