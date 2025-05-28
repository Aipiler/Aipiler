from __future__ import annotations
import gc
from typing import Callable, Dict, List, Optional, Union, Tuple
from collections import defaultdict

import Aipiler

import Aipiler.runtime

import Aipiler.runtime.device
from .device import Device, to_device


def nbytes2str(nbytes: int) -> str:
    if nbytes > 1024 * 1024:
        size = nbytes // 1024 // 1024
        unit = "MiB"
    elif nbytes > 1024:
        size = nbytes // 1024
        unit = "KiB"
    else:
        size = nbytes
        unit = "Bytes"
    return "{} {}".format(size, unit)


class MemoryAPI:
    def __init__(self, device: Device):
        self.device: Device = device
        self.addr2nbytes: Dict[int, int] = {}
        self.peak_allocated: int = 0
        self.allocated: int = 0

    def malloc(self, nbytes: int) -> int:
        raise NotImplementedError

    def free(self, addr: int):
        raise NotImplementedError

    def memory_info(self) -> Tuple[int, int]:
        raise NotImplementedError


class CpuMemoryAPI(MemoryAPI):
    def malloc(self, nbytes: int) -> int:
        from Aipiler.ffi import cruntime

        addr = cruntime.malloc(nbytes)
        if addr == 0 and nbytes != 0:
            return 0
        self.allocated += nbytes
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr: int):
        from Aipiler.ffi import cruntime

        cruntime.free(addr)
        self.allocated -= self.addr2nbytes.pop(addr)

    def memory_info(self) -> Tuple[int, int]:
        raise NotImplementedError()


class MemoryPool:
    def __init__(self, memory_api: MemoryAPI, block_size: int, max_reserve_size: int):
        self.memory_api: MemoryAPI = memory_api
        self.block_size: int = block_size
        self.max_reserve_size: int = max_reserve_size
        self.reserved_size: int = 0
        self.active_blocks = 0
        self.memory_blocks: Dict[int, List[Storage]] = defaultdict(list)

    def malloc(self, nbytes: int) -> Storage:
        allocated = (nbytes + self.block_size - 1) // self.block_size * self.block_size
        block_list = self.memory_blocks[allocated]
        if len(block_list) > 0:
            storage = block_list.pop()
            addr = storage.addr
            self.reserved_size -= storage.num_bytes
        else:
            addr = self.memory_api.malloc(allocated)
            if addr == 0 and allocated != 0:
                # out of memory
                gc.collect()
                self.clear()
                addr = self.memory_api.malloc(allocated)
                if addr == 0:
                    raise MemoryError(
                        f"Can not allocate {nbytes2str(allocated, True)} from {self.memory_api.device} device. "
                        + self.status(color=True)
                    )
        return Storage(
            device=self.memory_api.device,
            addr=addr,
            num_bytes=allocated,
            free_handler=self.free,
        )

    def free(self, storage: Storage):
        self.memory_blocks[storage.num_bytes].append(storage)
        self.reserved_size += storage.num_bytes
        if self.reserved_size > self.max_reserve_size:
            self.clear()

    def clear(self):
        # TODO: cuda sync
        for block_list in self.memory_blocks.values():
            for storage in block_list:
                self.memory_api.free(storage.addr)
                storage.addr = 0
        self.memory_blocks.clear()
        self.reserved_size = 0

    def status(self) -> str:
        allocated = self.memory_api.allocated
        peak_allocated = self.memory_api.peak_allocated
        items = [
            ["Allocated", allocated],
            ["Peak", peak_allocated],
            ["Reserved", self.reserved_size],
            ["Active", allocated - self.reserved_size],
        ]
        lines = [
            "Status of {} memory pool".format(self.memory_api.device),
            *["{:>12}: {}".format(name, nbytes2str(nbytes)) for name, nbytes in items],
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.status(color=True)

    def __del__(self):
        self.clear()


class MemoryPoolContext:
    def __init__(self, pool: MemoryPool):
        self.device: Device = pool.memory_api.device
        self.memory_pool: MemoryPool = pool
        self.prev_memory_pool: Optional[MemoryPool] = None

    def __enter__(self):
        self.prev_memory_pool = _device2pool[self.device]
        _device2pool[self.device] = self.memory_pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        _device2pool[self.device] = self.prev_memory_pool


class DeviceMemoryPools:
    def __init__(self):
        self.device2pool: Dict[Device, MemoryPool] = {}

    def __getitem__(self, device: Device) -> MemoryPool:
        if device not in self.device2pool:
            if device.is_cpu():
                self.device2pool[device] = MemoryPool(
                    CpuMemoryAPI(device),
                    block_size=4096,
                    max_reserve_size=512 * 1024**2,
                )
            else:
                raise ValueError("Unsupported device: {}".format(device))
        return self.device2pool[device]

    def __setitem__(self, device: Device, pool: MemoryPool):
        self.device2pool[device] = pool


_device2pool: DeviceMemoryPools = DeviceMemoryPools()


class Storage:
    def __init__(
        self,
        device: Device,
        addr: int,
        num_bytes: int,
        free_handler: Callable[[Storage], None],
    ):
        self.device: Device = device
        self.addr: int = addr
        self.num_bytes: int = num_bytes
        self.free_handler: Callable[[Storage], None] = free_handler

    def __del__(self):
        if self.addr != 0:
            self.free_handler(self)

    def __getstate__(self):
        raise ValueError()

    def __setstate__(self, state):
        raise ValueError()

    @staticmethod
    def new(device: Union[Device, str], num_bytes: int) -> Storage:
        """
        Allocate a new storage on the given device.
        """
        if isinstance(device, str):
            device = Aipiler.runtime.device.to_device(device)
        else:
            if not isinstance(device, Device):
                raise TypeError(
                    "device must be Device or str, but got {}".format(type(device))
                )
        memory_pool = _device2pool[device]
        return memory_pool.malloc(num_bytes)
