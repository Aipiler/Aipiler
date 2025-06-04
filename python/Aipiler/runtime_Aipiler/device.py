from typing import Optional


class Device:
    def __init__(self, device_type: str, device_index: Optional[int] = None):
        self.kind: str = device_type
        self.id: Optional[int] = device_index

    def __repr__(self) -> str:
        if self.id is None:
            return "device({})".format(self.kind)
        return "device({}, {})".format(self.kind, self.id)

    def __str__(self) -> str:
        if self.id is None:
            return self.kind
        return "{}:{}".format(self.kind, self.id)

    def __eq__(self, other):
        if not isinstance(other, Device):
            raise ValueError("Cannot compare Device with {}".format(type(other)))
        return self.kind == other.kind and self.id == other.id

    def __hash__(self):
        return hash((self.kind, self.id))

    def is_cpu(self) -> bool:
        return self.kind == "cpu"

    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def is_vcuda(self) -> bool:
        return self.kind == "vcuda"

    @property
    def target(self) -> str:
        return "cuda" if self.kind in ["cuda", "vcuda"] else "cpu"


def to_device(device_type: str, device_index: Optional[int] = None):
    if ":" in device_type:
        if device_index is not None:
            raise RuntimeError(
                'device_type must not contain ":" to specify device_index when device_index is '
                f"specified explicitly as a separate argument: ({device_type}, {device_index})"
            )
        items = device_type.split(":")
        if len(items) != 2:
            raise ValueError(f"Invalid device_type: {device_type}")
        device_type, device_index = items
        if not device_index.isdigit():
            raise ValueError(f"Invalid device_index: {device_index}")
        device_index = int(device_index)

    if device_type not in ["cpu", "cuda", "vcuda"]:
        raise ValueError(
            f'Invalid device_type: {device_type}, must be "cpu" "cuda" or "vcuda"'
        )

    if device_index is not None and not isinstance(device_index, int):
        raise ValueError(f"Invalid device_index: {device_index}, must be an integer")

    return Device(device_type, device_index)
