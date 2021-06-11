# Course: CS261 - Data Structures
# Author: Ian Fonberg
# Assignment: Graph Implementation (Portfolio Assignment)
# Description: DirectedGraph with implementations for add_vertex(), add_edge(),
#              remove_edge(), get_vertices(), get_edges(), is_valid_path(),
#              dfs(), bfs(), has_cycle(), and dijkstra().

import heapq
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """Adds vertices to the graph in ascending order.

        :return: number of vertices in the graph
        """
        self.v_count += 1
        self.adj_matrix.append([0] * self.v_count)
        for row in range(self.v_count - 1):
            # Add another column to each other vertex.
            self.adj_matrix[row].append(0)

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """Adds an edge to the graph from src to dst. If either of the
        vertices are not in the graph, src is dst or weight is not positive,
        then the method does nothing.

        :param src: head vertex
        :param dst: tail vertex
        :param weight: edge weight
        :return: nothing
        """
        rows = range(self.v_count)
        if (
                src not in rows or dst not in rows or
                weight <= 0 or src == dst
        ):
            # Vertices not in graph or equal themselves or weight is not
            # positive.
            return

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """Removes an edge between src and dst. If either src or dst is not
        in the graph or if there is no edge between them, the method does
        nothing.

        :param src: head vertex
        :param dst: tail vertex
        :return: nothing
        """
        rows = range(self.v_count)
        if src not in rows or dst not in rows:
            # Vertices not in graph.
            return

        # If there is no edge, the weight will be 0 anyway.
        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """Returns a list of vertices of the graph.

        :return: list of vertices
        """
        return list(range(self.v_count))

    def get_edges(self) -> []:
        """Returns a list of edges in the graph. Each edge is represented by a
        tuple of incident vertices along with the weight of the edge. First
        element is the source, second element is the destination, and third
        element is the weight.

        :return: list of edges
        """
        edges = set()

        for src in range(self.v_count):
            # Iterate over each out vertex.
            for dst in range(len(self.adj_matrix[src])):
                weight = self.adj_matrix[src][dst]
                if weight > 0:
                    # If an edge exists, add it.
                    edges.add((src, dst, weight))

        return list(edges)

    def is_valid_path(self, path: []) -> bool:
        """Takes a list of vertex names and returns True if the sequence of
        vertices represents a valid path in the graph and False otherwise.
        Empty paths are considered valid.

        :param path: list of vertices
        :return: true or false
        """
        # One vertex in path, check if it's in the graph.
        if len(path) == 1:
            return 0 <= path[0] <= self.v_count

        # Iterate over path vertices and check that the destination is in the
        # source's columns.
        for v_index in range(len(path) - 1):
            src = path[v_index]
            dst = path[v_index + 1]
            if (
                    src not in range(self.v_count) or
                    dst not in range(self.v_count) or
                    self.adj_matrix[src][dst] == 0
            ):
                # src or dst not in graph or no edge exists between them.
                return False

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """Return list of vertices visited during DFS search; vertices are
        picked in ascending order. If v_end is not provided it will search
        the entire graph. Otherwise, the search will stop when v_end is
        reached. If the starting vertex is not in the graph, nothing happens.


        :param v_start: search start vertex
        :param v_end: search end vertex or nothing
        :return: list of visited vertices
        """
        seen = []  # Use a list to preserve visited order.
        if v_start not in range(self.v_count):
            # Start vertex not in graph.
            return seen

        stack = deque()
        stack.append(v_start)  # Push start to the stack to begin search.
        while len(stack) > 0:
            vertex = stack.pop()
            if vertex in seen:
                continue
            else:
                seen.append(vertex)
                if vertex == v_end:
                    # Target vertex found; end search.
                    break
                out_vertices = self.adj_matrix[vertex]
                # Push vertices to stack in ascending order.
                for out_vertex in range(len(out_vertices) - 1, -1, -1):
                    if out_vertex not in seen and out_vertices[out_vertex] != 0:
                        stack.append(out_vertex)

        return seen

    def bfs(self, v_start, v_end=None) -> []:
        """Return list of vertices visited during BFS search; vertices are
        picked in ascending order. If v_end is not provided it will search
        the entire graph. Otherwise, the search will stop when v_end is
        reached. If the starting vertex is not in the graph, nothing happens.

        :param v_start: search start vertex
        :param v_end: search end vertex
        :return: list of visited vertices
        """
        seen = []  # Use a list to preserve visited order.
        if v_start not in range(self.v_count):
            # Start vertex not in graph.
            return seen

        queue = deque()
        queue.append(v_start)
        while len(queue) > 0:
            vertex = queue.popleft()
            if vertex in seen:
                continue
            else:
                seen.append(vertex)
                if vertex == v_end:
                    # Target vertex found; end search.
                    break
                out_vertices = self.adj_matrix[vertex]
                # Push vertices to stack in ascending order.
                for out_vertex in range(len(out_vertices)):
                    if out_vertex not in seen and out_vertices[out_vertex] != 0:
                        queue.append(out_vertex)

        return seen

    def _rec_has_cycle(self, vertex, seen, cycle):
        """Uses recursion to perform a DFS so that it can keep track of the
        vertices that the current vertex has seen. If current vertex has already
        been seen in the current vertex's search, cycle will be set to True.
        Otherwise, it remains False.

        :param vertex: current vertex
        :param seen: seen vertices for the current vertex
        :param cycle: graph contains a cycle
        :return: true or false
        """
        if vertex in seen:
            # Vertex has already been seen; cycle exists.
            return True
        else:
            # Add visited vertex to seen.
            seen.add(vertex)

        num_out_vertices = len([v for v in self.adj_matrix[vertex] if v != 0])
        for out_vertex in range(self.v_count):
            if self.adj_matrix[vertex][out_vertex] > 0:
                if num_out_vertices > 1:
                    # Generate a new copy of the set to avoid conflicts with
                    # other searches.
                    current_seen = seen.copy()
                else:
                    current_seen = seen
                cycle = self._rec_has_cycle(out_vertex, current_seen, cycle)

        # Return result of recursive call.
        return cycle

    def has_cycle(self):
        """Returns True if there is at least one cycle in the graph.

        :return: true or false
        """
        for v_start in range(self.v_count):
            # Test each vertex for cycle using DFS.
            # Recursion makes it easier to keep track of visited vertices.
            if self._rec_has_cycle(v_start, set(), False):
                return True

        return False

    def dijkstra(self, src: int) -> []:
        """Implements the Dijkstra algorithm to compute the length of the
        shortest path from a given vertex to all other vertices in the graph.
        Returns a list with one value per each vertex in the graph, where the
        value represents the distance between the src and that vertex.

        :param src: start vertex
        :return: list of vertices and their length from src.
        """
        visited = [float('inf')] * self.v_count
        priority = []
        # Use min heap to remove current shortest distance.
        heapq.heappush(priority, (0, src))
        while len(priority) > 0:
            vertex = heapq.heappop(priority)  # Pop minimum vertex.
            if visited[vertex[1]] == float('inf'):
                # If the vertex is the shortest, add it to visited.
                visited[vertex[1]] = vertex[0]
                # Add out vertices to the min heap.
                for out_vertex in range(self.v_count):
                    value = self.adj_matrix[vertex[1]][out_vertex]
                    if value != 0:
                        heapq.heappush(priority, (vertex[0] + value,
                                                  out_vertex))

        return visited


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3, 20], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
