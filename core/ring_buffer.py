"""Lock-free shared memory ring buffer using multiprocessing.shared_memory.

Design notes:
- Single writer and single reader are assumed for cursor integrity.
- Metadata layout: int64 head, int64 tail (monotonic counters, not wrapped).
- Each slot stores a 4-byte little-endian length prefix + payload bytes.
"""
from multiprocessing import shared_memory
from typing import Optional

import numpy as np


class SharedMemoryRingBuffer:
    METADATA_SIZE = 16  # int64 head + int64 tail
    LENGTH_PREFIX_SIZE = 4

    def __init__(self, name: str, size: int, slot_size: int, create: bool = False, overwrite: bool = False) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if slot_size <= self.LENGTH_PREFIX_SIZE:
            raise ValueError("slot_size must exceed length prefix")

        self.name = name
        self.size = size
        self.slot_size = slot_size
        self.overwrite = overwrite
        self._total_bytes = self.METADATA_SIZE + (size * slot_size)

        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self._total_bytes)
            self._meta = np.ndarray((2,), dtype=np.int64, buffer=self.shm.buf)
            self._meta[:] = 0
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            if self.shm.size < self._total_bytes:
                raise ValueError("existing shared memory is smaller than required")
            self._meta = np.ndarray((2,), dtype=np.int64, buffer=self.shm.buf)

        self._buffer_view = self.shm.buf[self.METADATA_SIZE : self._total_bytes]

    def write(self, data: bytes) -> None:
        payload_len = len(data)
        if payload_len > (self.slot_size - self.LENGTH_PREFIX_SIZE):
            raise ValueError("data too large for slot")

        head = int(self._meta[0])
        tail = int(self._meta[1])
        if head - tail >= self.size:
            if not self.overwrite:
                raise BufferError("ring buffer is full")
            tail += 1
            self._meta[1] = tail

        idx = head % self.size
        offset = idx * self.slot_size
        slot = self._buffer_view[offset : offset + self.slot_size]
        slot[: self.LENGTH_PREFIX_SIZE] = payload_len.to_bytes(self.LENGTH_PREFIX_SIZE, byteorder="little", signed=False)
        slot[self.LENGTH_PREFIX_SIZE : self.LENGTH_PREFIX_SIZE + payload_len] = data
        # Zero-fill the remaining bytes to avoid stale data reads when reusing slots.
        if payload_len < (self.slot_size - self.LENGTH_PREFIX_SIZE):
            slot[self.LENGTH_PREFIX_SIZE + payload_len : self.slot_size] = b"\x00" * (
                self.slot_size - self.LENGTH_PREFIX_SIZE - payload_len
            )

        self._meta[0] = head + 1

    def read(self) -> Optional[bytes]:
        head = int(self._meta[0])
        tail = int(self._meta[1])
        if head == tail:
            return None

        idx = tail % self.size
        offset = idx * self.slot_size
        slot = self._buffer_view[offset : offset + self.slot_size]
        payload_len = int.from_bytes(slot[: self.LENGTH_PREFIX_SIZE], byteorder="little", signed=False)
        if payload_len < 0 or payload_len > (self.slot_size - self.LENGTH_PREFIX_SIZE):
            raise ValueError("invalid payload length in slot")

        data = bytes(slot[self.LENGTH_PREFIX_SIZE : self.LENGTH_PREFIX_SIZE + payload_len])
        self._meta[1] = tail + 1
        return data

    def close(self, unlink: bool = False) -> None:
        self.shm.close()
        if unlink:
            self.shm.unlink()

    def __enter__(self) -> "SharedMemoryRingBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
