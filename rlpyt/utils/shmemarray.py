
"""
Currently unused in rlpyt, but might be useful later.  For creating shared
memory between processes AFTER they fork.
"""

#
# Based on multiprocessing.sharedctypes.RawArray
#
# Uses posix_ipc (http://semanchuk.com/philip/posix_ipc/) to allow shared ctypes arrays
# among unrelated processors
#
# Usage Notes:
#    * The first two args (typecode_or_type and size_or_initializer) should work the same as with RawArray.
#    * The shared array is accessible by any process, as long as tag matches.
#    * The shared memory segment is unlinked when the origin array (that returned
#      by ShmemRawArray(..., create=True)) is deleted/gc'ed
#    * Creating an shared array using a tag that currently exists will raise an ExistentialError
#    * Accessing a shared array using a tag that doesn't exist (or one that has been unlinked) will also
#      raise an ExistentialError
#
# Author: Shawn Chin (http://shawnchin.github.com)
#
# Edited for python 3 by: Adam Stooke
#

import numpy as np
# import os
import time
import sys
import mmap
import ctypes
import posix_ipc
# from _multiprocessing import address_of_buffer  # (not in python 3)
from string import ascii_letters, digits

valid_chars = frozenset("/-_. %s%s" % (ascii_letters, digits))

typecode_to_type = {
    'c': ctypes.c_char, 'u': ctypes.c_wchar,
    'b': ctypes.c_byte, 'B': ctypes.c_ubyte,
    'h': ctypes.c_short, 'H': ctypes.c_ushort,
    'i': ctypes.c_int, 'I': ctypes.c_uint,
    'l': ctypes.c_long, 'L': ctypes.c_ulong,
    'f': ctypes.c_float, 'd': ctypes.c_double
}


def address_of_buffer(buf):  # (python 3)
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class ShmemBufferWrapper:

    def __init__(self, tag, size, create=True):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create
        self.size = size

        assert 0 <= size < sys.maxsize  # sys.maxint  (python 3)
        flag = (0, posix_ipc.O_CREX)[create]
        self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
        self._map = mmap.mmap(self._mem.fd, self._mem.size)
        self._mem.close_fd()

    def get_address(self):
        # addr, size = address_of_buffer(self._map)
        # assert size == self.size
        assert self._map.size() == self.size  # (changed for python 3)
        addr = address_of_buffer(self._map)
        return addr

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()


def ShmemRawArray(typecode_or_type, size_or_initializer, tag, create=True):
    assert frozenset(tag).issubset(valid_chars)
    if tag[0] != "/":
        tag = "/%s" % (tag,)

    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
    else:
        type_ = type_ * len(size_or_initializer)

    buffer = ShmemBufferWrapper(tag, ctypes.sizeof(type_), create=create)
    obj = type_.from_address(buffer.get_address())
    obj._buffer = buffer

    if not isinstance(size_or_initializer, int):
        obj.__init__(*size_or_initializer)

    return obj


###############################################################################
#                       New Additions  (by Adam)                              #


def NpShmemArray(shape, dtype, tag, create=True):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    shmem = ShmemRawArray(ctypes.c_char, nbytes, tag, create)
    return np.frombuffer(shmem, dtype=dtype, count=size).reshape(shape)


def get_random_tag():
    return str(time.time()).replace(".", "")[-9:]
