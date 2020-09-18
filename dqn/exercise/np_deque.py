import numpy as np


class np_deque:
    """
    A FIFO circular storage based on numpy array that:
      - allows fast numpy indexing,
      - has a maximum size,
      - before the buffer if filled acts as if its length is as many elements as were appended so far
    """

    def __init__(self, buffer_size, fields_size):
        """
        :param buffer_size: max length of a queue
        :type buffer_size: int
        :param fields_size: fixed length of each element
        :type fields_size: int
        """
        self._queue = np.zeros((buffer_size, fields_size), dtype=np.float)
        self._last = 0
        self._filled = False

    def from_array(self, arr):
        """
        Overwrites the queue with the provided array
        :param arr: array to load from, should be shorter or equal to buffer_size
        :type arr: numpy.ndarray
        """
        assert arr.shape[0] <= self._queue.shape[0], 'Provided array to load from is bigger than max_size'
        assert arr.shape[1] == self._queue.shape[1], 'Provided array has a wrong number of fields (arr.shape[1])'

        self._queue.fill(0)
        self._queue[:arr.shape[0]] = arr
        if arr.shape[0] == self._queue.shape[0]:
            self._filled = True
            self._last = 0
        else:
            self._last = arr.shape[0]

    def append(self, el):
        """
        Adds an element into the queue into a vacant place if available, overrides the oldest element otherwise
        :param el: element to add, must have  length of fields_size
        :type el: list
        """
        self._queue[self._last] = el
        self._last += 1

        if self._last == self._queue.shape[0]:
            self._filled = True
            self._last = 0

    def max_size(self):
        """
        :return: max buffer size
        :rtype: int
        """
        return self._queue.shape[0]

    def __len__(self):
        """
        :return: number of currently stored elements
        :rtype: i t
        """
        if self._filled:
            return self._queue.shape[0]
        else:
            return self._last

    def __bool__(self):
        return len(self) > 0

    def __getitem__(self, key):
        """
        Overrides the slicing of numpy array to make ":" slice end with the number of currently stored elements
        :param key:
        :type key:
        :return: A picked slice
        :rtype: numpy.ndarray
        """
        if not self._filled:
            if isinstance(key, tuple) and isinstance(key[0], slice) and key[0].stop is None:
                key = (slice(key[0].start, self._last, key[0].step), key[1])
            elif isinstance(key, slice) and (key.stop is None or key.stop == np.iinfo(int).max):
                key = slice(key.start, self._last, key.step)
        return self._queue[key]

    def __setitem__(self, key, value):
        self._queue[key] = value


if __name__ == '__main__':
    # Create an empty np_deque
    memory = np_deque(5, 8)
    print('The queue now has {} elements'.format(len(memory)))
    print('The queue can store up to {} elements'.format(memory.max_size()))
    print(memory == True)
    print('\n')

    # Fill np_deque
    for i in range(10):
        memory.append(list(range(i, 8 + i)))
        print('The queue now has {} element'.format(len(memory)) + ('s' if i > 0 else ''))
        print('The filled part of the queue:')
        print(memory[:])

    # Replace with a new array
    print('\n')
    print('Overriding the queue with a new array 3x8:')
    memory.from_array(np.ones((3, 8)))
    print(memory[:])