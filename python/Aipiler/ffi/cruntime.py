import ctypes
from typing import Optional


class LibCAPI:
    def __init__(self):
        self.libc = ctypes.CDLL("libc.so.6")

    def malloc(self, nbytes: int) -> int:
        """
        Allocate cpu memory.

        Parameters
        ----------
        size: int
            The number of bytes to allocate.

        Returns
        -------
        ret: int
            The pointer to the allocated memory.
        """
        self.libc.malloc.argtypes = [ctypes.c_size_t]
        self.libc.malloc.restype = ctypes.c_void_p
        return int(self.libc.malloc(nbytes))

    def free(self, addr: int) -> None:
        """
        Free cpu memory.

        Parameters
        ----------
        addr: int
            The pointer to the memory to be freed. The pointer must be returned by malloc.
        """
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        return self.libc.free(addr)


_LIBCAPI: Optional[LibCAPI] = None


def lazy_load_libc() -> None:
    global _LIBCAPI
    if _LIBCAPI:
        return
    try:
        _LIBCAPI = LibCAPI()
    except OSError as e:
        print("Failed to load C runtime library.")
        raise e


# expose libc_malloc and libc_free to storage
def malloc(size: int) -> int:
    lazy_load_libc()
    return _LIBCAPI.malloc(size)


def free(ptr: int) -> None:
    lazy_load_libc()
    _LIBCAPI.free(ptr)
