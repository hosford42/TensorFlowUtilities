from typing import Tuple, Optional

import tensorflow as tf


class CyclicTensorBuffer:
    """An efficient deque-like container for tensors. Useful for accumulating training data gathered
    during execution of on-line algorithms such as reinforcement learning."""

    def __init__(self, record_shape: Tuple[int, ...], capacity: int = None, dtype=tf.float32):
        if capacity is None:
            capacity = 1
            dynamic_capacity = True
        else:
            dynamic_capacity = False
        self._record_shape = tuple(record_shape)
        self._capacity = capacity  # Maximum number of records that can be stored.
        self._dynamic_capacity = dynamic_capacity  # Whether to auto-adjust capacity as needed.
        self._start = 0  # Offset of first filled record in array
        self._size = 0  # Number of records stored in array
        self._array = tf.Variable(
            tf.zeros((self._capacity or 0,) + self._record_shape, dtype=dtype),
            shape=(None,) + self._record_shape,
            dtype=dtype
        )

    @property
    def record_shape(self) -> Tuple[int, ...]:
        """The shape of the records this buffer holds."""
        return self._record_shape

    @property
    def capacity(self) -> Optional[int]:
        """The maximum number of records the buffer is currently configured to hold."""
        if self._dynamic_capacity:
            return None
        return self._capacity

    @property
    def size(self) -> int:
        """The number of records the buffer currently holds."""
        return self._size

    @property
    def dynamic_capacity(self) -> bool:
        """Whether the buffer will automatically adjust capacity as needed."""
        return self._dynamic_capacity

    @dynamic_capacity.setter
    def dynamic_capacity(self, value: bool) -> None:
        self._dynamic_capacity = bool(value)

    def set_capacity(self, capacity: Optional[int], delete: bool = False) -> None:
        self._dynamic_capacity = capacity is None
        if self._dynamic_capacity:
            return
        assert capacity is not None
        self._resize_buffer(capacity, delete)

    def _resize_buffer(self, capacity: int, delete: bool = False) -> None:
        if capacity < self._size and not delete:
            raise IndexError("Buffer would have insufficient capacity for its current contents.")
        values = self.stack()[-capacity:]
        empty = tf.zeros((capacity - self._size,) + self._record_shape, dtype=values.dtype)
        self._array.assign(tf.concat([values, empty], axis=0))
        self._start = 0
        self._capacity = capacity

    def peek(self) -> tf.Tensor:
        if self._size <= 0:
            raise IndexError("Buffer is empty.")
        return self._array[(self._start + self._size - 1) % self._capacity]

    def peekleft(self) -> tf.Tensor:
        if self._size <= 0:
            raise IndexError("Buffer is empty.")
        return self._array[self._start]

    def append(self, value, overwrite: bool = False) -> None:
        value = tf.convert_to_tensor(value)
        if self._size >= self._capacity:
            if self._dynamic_capacity:
                self._resize_buffer(self._capacity * 2)
            elif overwrite:
                self.popleft()
            else:
                raise IndexError("Buffer is full.")
        index = (self._start + self._size) % self._capacity
        self._array.scatter_nd_update([index], [value])
        self._size += 1

    def appendleft(self, value, overwrite: bool = False) -> None:
        value = tf.convert_to_tensor(value)
        if self._size >= self._capacity:
            if self._dynamic_capacity:
                self._resize_buffer(self._capacity * 2)
            elif overwrite:
                self.pop()
            else:
                raise IndexError("Buffer is full.")
        index = (self._start - 1) % self._capacity
        self._array.scatter_nd_update([index], [value])
        self._start = index
        self._size += 1

    def pop(self) -> tf.Tensor:
        if self._size <= 0:
            raise IndexError("Buffer is empty.")
        index = (self._start + self._size - 1) % self._capacity
        value = self._array[index]
        self._size -= 1
        return value

    def popleft(self) -> tf.Tensor:
        if self._size <= 0:
            raise IndexError("Buffer is empty.")
        index = self._start
        value = self._array[index]
        self._start = (self._start + 1) % self._capacity
        self._size -= 1
        return value

    def stack(self) -> tf.Tensor:
        end = self._start + self._size
        if end <= self._capacity:
            return self._array[self._start:end]
        else:
            end = end % self._capacity
            return tf.concat([self._array[self._start:], self._array[:end]], axis=0)

    def clear(self) -> None:
        self._start = 0
        self._size = 0
