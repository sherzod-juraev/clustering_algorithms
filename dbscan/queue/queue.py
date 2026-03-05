from .node import Node
from .exception import Empty
from typing import Any


class Queue:
    """
    A simple FIFO (First-In-First-Out) queue implemented using a singly linked list.
    """

    def __init__(self):
        """
        Initialize an empty queue.
        """

        self.__head = None
        self.__tail = None
        self.__length = 0

    def __autoincrement_length(self):
        """Increment the internal length counter"""

        self.__length += 1

    def __autodecrement_length(self):
        """Decrement the internal length counter"""
        self.__length -= 1

    def __normalize_index(self, index: int, /) -> int:
        """
        Normalize a possibly negative index to a valid positive index.

        Args:
            index (int): The index to normalize.

        Returns:
            int: Normalized index.

        Raises:
            IndexError: If index is out of range.
        """

        index = index if index >= 0 else self.__length + index
        if self.__length <= index or index < 0:
            raise IndexError('Index out of range')
        return index

    def __verify_empty(self):
        """
        Raise an exception if the queue is empty.

        Raises:
            Empty: If the queue has no elements.
        """

        if self.__length == 0:
            raise Empty('Queue is empty')

    def __len__(self):
        """
        Return the number of elements in the queue.

        Returns:
            int: Queue length.
        """

        return self.__length

    def __iter__(self):
        """
        Return an iterator over the queue elements in FIFO order.

        Yields:
            Any: Next element in the queue.
        """

        current = self.__head
        while current:
            yield current.value
            current = current.next

    def __getitem__(self, item):
        """
        Return the element at the specified index.

        Args:
            item (int): Index of the element.

        Returns:
            Any: Element at the given index.

        Raises:
            Empty: If the queue is empty.
            IndexError: If index is out of range.
        """

        self.__verify_empty()
        item = self.__normalize_index(item)
        current = self.__head
        for _ in range(item):
            current = current.next
        return current.value

    def __setitem__(self, key, value):
        """
        Set the element at the specified index to a new value.

        Args:
            key (int): Index of the element.
            value (Any): New value to assign.

        Raises:
            Empty: If the queue is empty.
            IndexError: If index is out of range.
        """

        self.__verify_empty()
        key = self.__normalize_index(key)
        current = self.__head
        for _ in range(key):
            current = current.next
        current.value = value

    def enqueue(self, value: Any, /):
        """
        Add an element to the end of the queue.

        Args:
            value (Any): The element to add.
        """

        if self.__length == 0:
            self.__head = Node(value)
            self.__tail = self.__head
            self.__autoincrement_length()
            return
        self.__tail.next = Node(value)
        self.__tail = self.__tail.next
        self.__autoincrement_length()

    def dequeue(self) -> Any:
        """
        Remove and return the element from the front of the queue.

        Returns:
            Any: The removed element.

        Raises:
            Empty: If the queue is empty.
        """

        self.__verify_empty()
        current = self.__head
        self.__head = self.__head.next
        self.__autodecrement_length()
        return current.value

    def peek(self):
        """
        Return the element at the front of the queue without removing it.

        Returns:
            Any: The front element.

        Raises:
            Empty: If the queue is empty.
        """

        self.__verify_empty()
        return self.__head.value

    def add_list(self, data: list) -> None:

        for item in data:
            self.enqueue(item)