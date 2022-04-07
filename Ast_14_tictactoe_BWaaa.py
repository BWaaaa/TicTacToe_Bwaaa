# -*- coding: utf-8 -*-*
"""
Tic Tac Toe Player
"""
__author__ = 'Brian Wang'
__date__ = 'Apr 6 2022'


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    turn = 0
    for i in board:
        for j in i:
            if j == X:
                turn += 1
            elif j == O:
                turn -= 1
    return "X" if turn <= 0 else "O"
    # raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    act = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
               act.add((i,j))
    return act


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    pla = player(board)
    new_board = [[board[i][j] for j in range(3)] for i in range(3)]
    # a new copy of the board
    new_board[i][j] = pla
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Assuming the board is now full
    for i in board:
        if i == [X,X,X]:
            return X
        elif i == [O,O,O]:
            return O
    for j in range(3):
        col = [row[j] for row in board]
        if col == [X,X,X]:
            return X
        elif col == [O, O, O]:
            return O
    diag1 = [board[d][d] for d in range(3)]
    diag2 = [board[d][2-d] for d in range(3)]
    if diag1 == [X,X,X] or diag2 == [X,X,X]:
        return X
    if diag1 == [O, O, O] or diag2 == [O, O, O]:
        return O


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    a = winner(board)
    if a is not None:
        return True
    # return "DRAW" if actions(board) == set() else False
    return True if actions(board) == set() else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    a = winner(board)
    if a == X:
        return 1
    if a == O:
        return -1
    return 0


def summary(board):
    xer = 0
    oer = 0
    for i in board:
        for j in i:
            if j == X:
                xer += 1
            elif j == O:
                oer += 1
    return X+str(xer)+O+str(oer)


def actionfinder(board, next_b):
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY and next_b[i][j] != EMPTY:
                return (i,j)


class LinkedQueue:
    """FIFO queue implementation using a singly linked list for storage."""

    #-------------------------- nested _Node class --------------------------
    class _Node:
        """Lightweight, nonpublic class for storing a singly linked node."""
        __slots__ = '_element', '_next'         # streamline memory usage

        def __init__(self, element, next):
            self._element = element
            self._next = next

    #------------------------------- queue methods -------------------------------
    def __init__(self):
        """Create an empty queue."""
        self._head = None
        self._tail = None
        self._size = 0                          # number of queue elements

    def __len__(self):
        """Return the number of elements in the queue."""
        return self._size

    def is_empty(self):
        """Return True if the queue is empty."""
        return self._size == 0

    def first(self):
        """Return (but do not remove) the element at the front of the queue.

        Raise Empty exception if the queue is empty.
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        return self._head._element              # front aligned with head of list

    def dequeue(self):
        """Remove and return the first element of the queue (i.e., FIFO).

        Raise Empty exception if the queue is empty.
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        if self.is_empty():                     # special case as queue is empty
            self._tail = None                     # removed head had been the tail
        return answer

    def enqueue(self, e):
        """Add an element to the back of queue."""
        newest = self._Node(e, None)            # node will be new tail node
        if self.is_empty():
            self._head = newest                   # special case: previously empty
        else:
            self._tail._next = newest
        self._tail = newest                     # update reference to tail node
        self._size += 1

    def __str__(self):
        result = []
        curNode = self._head
        while (curNode is not None):
            result.append(str(curNode._element) + " --> ")
            curNode = curNode._next
        result.append("None")
        return "".join(result)


class MultiTree_TTT:
    class TreeNode:
        def __init__(self, element = initial_state(), parent = None, children = None, utility = 0):
            self._parent = parent
            self._element = element
            self._children = children
            self._value = utility

        def __str__(self):
            # 这个思路是回答小朋友问题借鉴的好主意
            if not isinstance(self._element, list):
                return str(self._element)
            return str([summary(self._element), player(self._element), self._value])

    #-------------------------- binary tree constructor --------------------------
    def __init__(self):
        """Create an initially empty binary tree."""
        self._root = None
        self._size = 0

    #-------------------------- public accessors ---------------------------------
    def __len__(self):
        """Return the total number of elements in the tree."""
        return self._size

    def is_root(self, node):
        """Return True if a given node represents the root of the tree."""
        return self._root == node

    def is_leaf(self, node):
        """Return True if a given node does not have any children."""
        return self.num_children(node) == 0

    def is_empty(self):
        """Return True if the tree is empty."""
        return len(self) == 0

    def __iter__(self):
        """Generate an iteration of the tree's elements."""
        for node in self.nodes():                        # use same order as nodes()
            yield node._element                               # but yield each element

    def nodes(self):
        """Generate an iteration of the tree's nodes."""
        return self.postorder()                            # return entire postorder iteration

    def preorder(self):
        """Generate a preorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_preorder(self._root):  # start recursion
                yield node

    def _subtree_preorder(self, node):
        """Generate a preorder iteration of nodes in subtree rooted at node."""
        yield node                                           # visit node before its subtrees
        for c in self.children(node):                        # for each child c
            for other in self._subtree_preorder(c):         # do preorder of c's subtree
                yield other                                   # yielding each to our caller

    def postorder(self):
        """Generate a postorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_postorder(self._root):  # start recursion
                yield node

    def _subtree_postorder(self, node):
        """Generate a postorder iteration of nodes in subtree rooted at node."""
        for c in self.children(node):                        # for each child c
            for other in self._subtree_postorder(c):        # do postorder of c's subtree
                yield other                                   # yielding each to our caller
        yield node                                           # visit node after its subtrees

    def breadthfirst(self):
        """Generate a breadth-first iteration of the nodes of the tree."""
        if not self.is_empty():
            fringe = LinkedQueue()             # known nodes not yet yielded
            fringe.enqueue(self._root)        # starting with the root
            while not fringe.is_empty():
                node = fringe.dequeue()             # remove from front of the queue
                yield node                          # report this node
                for c in self.children(node):
                    fringe.enqueue(c)              # add children to back of queue

    def root(self):
        """Return the root of the tree (or None if tree is empty)."""
        return self._root

    def parent(self, node):
        """Return node's parent (or None if node is the root)."""
        return node._parent

    def children(self, node):
        """Generate an iteration of nodes representing node's children."""
        if node._children is None:      # use is to test for None
            return
        for each in node._children:
            yield each

    def num_children(self, node):
        """Return the number of children of a given node."""
        return len(node._children) if node._children is not None else 0

    def sibling(self, node):
        """Return a node representing given node's sibling (or None if no sibling)."""
        parent = node._parent
        if parent is None:                    # p must be the root
            return None                         # root has no sibling
        else:
            for each in self.children(parent):
                if each != node:
                    yield each

    #-------------------------- nonpublic mutators --------------------------
    def add_root(self, e):
        """Place element e at the root of an empty tree and return the root node.

        Raise ValueError if tree nonempty.
        """
        if self._root is not None:
            raise ValueError('Root exists')
        self._size = 1
        self._root = self.TreeNode(e, None, [])
        return self._root

    def add_child(self, node, e):
        """Create a new left child for a given node, storing element e in the new node.

        Return the new node.
        Raise ValueError if node already has a left child.
        """
        self._size += 1
        new_node = self.TreeNode(e, node, [])
        node._children.append(new_node)             # node is its parent
        return new_node

    def _replace(self, node, e):
        """Replace the element at given node with e, and return the old element."""
        old = node._element
        node._element = e
        return old

    def _delete(self, node):
        """Delete the given node, and replace it with its child, if any.

        Return the element that had been stored at the given node.
        Raise ValueError if node has two children.
        """
        if self.num_children(node) >= 2:
            raise ValueError('Position has more than one children')
        child = node._children[0] if len(node._children)>0 else None  # might be None
        if child is not None:
            child._parent = node._parent     # child's grandparent becomes parent
        if node is self._root:
            self._root = child             # child becomes root
        else:
            parent = node._parent
            if child is None:
                parent._children.remove(node)
            else:
                idx = parent._children.index(node)
                parent._children[idx] = child
        self._size -= 1
        return node._element

    def _attach(self, node, list_of_trees):
        """Attach trees t1 and t2, respectively, as the left and right subtrees of the external node.

        As a side effect, set t1 and t2 to empty.
        Raise TypeError if trees t1 and t2 do not match type of this tree.
        Raise ValueError if node already has a child. (This operation requires a leaf node!)
        """
        if not self.is_leaf(node):
            raise ValueError('position must be leaf')
        for each in list_of_trees:
            if not type(self) is type(each):    # all 3 trees must be same type
                raise TypeError('Tree types must match')
        self._size += sum([len(t1) for t1 in list_of_trees])
        for t1 in list_of_trees:      # attached t1 as left subtree of node
            t1._root._parent = node
            node._children.append(t1._root)
            t1._root = None             # set t1 instance to empty
            t1._size = 0
        return self

    def height(self, node=None):
       """Return the height of the subtree rooted at a given node.

       If node is None, return the height of the entire tree.
       """
       if node is None:
           node = self._root
       if self.is_leaf(node):
           return 0
       else:
           return 1 + max(self.height(c) for c in self.children(node))

    def depth(self, node):
        if self.is_root(node):
            return 0
        else:
            return 1 + self.depth(node._parent)

    # 先尽量保留，让它能用就行
    def pretty_print(self):
        # ----------------------- Need to enter height to work -----------------
        levels = self.height() + 1
        print("Levels:", levels)

        def print_internal(this_level_nodes, current_level, max_level):
            if (len(this_level_nodes) == 0 or all_elements_are_None(this_level_nodes)):
                return  # Base case of recursion: out of nodes, or only None left

            floor = max_level - current_level;
            endgeLines = 2 ** max(floor - 1, 0);
            firstSpaces = 2 ** floor - 1;
            betweenSpaces = 2 ** (floor + 1) - 1;
            print_spaces(firstSpaces)
            next_level_nodes = []
            for node in this_level_nodes:
                if (node is not None):
                    print(str(node), end="")
                    if node._children is not None:
                        for c in node._children:
                            next_level_nodes.append(c)
                        for _ in range(2-len(node._children)):
                            next_level_nodes.append(None)
                    else:
                        next_level_nodes.append(None)
                        next_level_nodes.append(None)
                else:
                    next_level_nodes.append(None)
                    next_level_nodes.append(None)
                    print_spaces(1)

                print_spaces(betweenSpaces)
            print()
            # for i in range(1, endgeLines + 1):
            #     for j in range(0, len(this_level_nodes)):
            #         print_spaces(firstSpaces - i)
            #         if (this_level_nodes[j] == None):
            #             print_spaces(endgeLines + endgeLines + i + 1);
            #             continue
            #         if (this_level_nodes[j]._left != None):
            #             print("/", end="")
            #         else:
            #             print_spaces(1)
            #         print_spaces(i + i - 1)
            #         if (this_level_nodes[j]._right != None):
            #             print("\\", end="")
            #         else:
            #             print_spaces(1)
            #         print_spaces(endgeLines + endgeLines - i)
            #     print()

            print_internal(next_level_nodes, current_level + 1, max_level)

        def all_elements_are_None(list_of_nodes):
            for each in list_of_nodes:
                if each is not None:
                    return False
            return True

        def print_spaces(number):
            for i in range(number):
                print(" ", end="")

        print_internal([self._root], 1, levels)

    def train(self):
        def board_forward(node):
            board = node._element
            # Start construct the game tree: forward
            if terminal(board):
                node._value = utility(board)
                # print('util', node._value)
                count[0] += 1
                if count[0] % 10000 == 0:
                    print('end game solution number', count[0])
                return
            for move in actions(board):
                # print(len(actions(board)))
                next_board = result(board, move)
                new_gameplay = self.add_child(node, next_board)
                board_forward(new_gameplay)

        def board_backward(node):
            board = node._element
            if terminal(board):
                return node._value
            if player(board) == X:
                myvalue = max([board_backward(child) for child in self.children(node)])
            else:
                myvalue = min([board_backward(child) for child in self.children(node)])
            node._value = myvalue
            return myvalue

        count = [0]
        # Now: pre-order traversal of adding board
        board_forward(self._root)
        print('my current tree has length', len(self))# ,'The max number of nodes is 9!=', math.factorial(9))
        print('my current tree has height', self.height(), 'Expected:', 9 - int(summary(self._root._element)[3]) - int(summary(self._root._element)[1]))
        print(count[0], "number of leaves: Expect <=255168 end-game solutions")
        # gt.pretty_print()
        # Now go backward, essentially is a postorder traverse
        board_backward(self._root)

    def board2node(self, board, node=None):
        if node == None:
            node = self._root
        if node._element == board:
            return node
        for child in self.children(node):
            i,j = actionfinder(node._element, child._element)
            if child._element[i][j] == board[i][j]:
                return self.board2node(board, child)
        raise ValueError("Board couldn't be found")


def minimax(board, gt):
    """
    Returns the optimal action for the current player on the board.

    #  We Can use this method: setting the board as root, and then look for solution
    gt.add_root(board)
    gt.train()
    # Advantage: only build part of the tree
    """
    mynode = gt.board2node(board)
    bestval = mynode._value
    bestresponse = set()
    for child in gt.children(mynode):
            if child._value == bestval:
                bestresponse.add(actionfinder(board, child._element))
    for each in bestresponse:
        return each  # just want to get one of them


def main_testing():
    gt = MultiTree_TTT()
    board = initial_state()
    # board = [[X, X, EMPTY], [O, O, X], [O, EMPTY ,X]]
    gt.add_root(board)
    gt.train()

    board = [[X, X, EMPTY], [O, O, X], [O, EMPTY ,X]]
    brp = minimax(board)
    mynode = gt.board2node(board)
    # print(len(c), '9!=', 9*8*7*6*5*4*3*2*1)
    # print(c.height())
    # gt.pretty_print()
    print(mynode ._element, mynode ._value)
    for child in gt.children(mynode ):
        print(child._value, child._element)
    print(brp)

    a = MultiTree_TTT()
    a.add_root(1)
    a.add_child(a._root, 3)
    a.add_child(a._root, 4)
    a.add_child(a._root, 5)
    a.add_child(a._root, 3)
    a.add_child(a._root, 9)
    a.add_child(a._root._children[1], 9)
    a.pretty_print()

    gt = MultiTree_TTT()
    gt.add_root(board)
    print(gt)
    gt.pretty_print()


main_testing()