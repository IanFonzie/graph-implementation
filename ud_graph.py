# Course: 
# Author: 
# Assignment: 
# Description:

from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """Add new vertex to the graph. If the vertex is already present in
        the graph nothing happens.

        :param v: vertex to add
        :return: nothing
        """
        if v in self.adj_list:
            return

        self.adj_list[v] = []
        
    def add_edge(self, u: str, v: str) -> None:
        """Adds a new edge between u and v to the graph. If u is v or an
        edge between u and v already exists in the graph, nothing happens. If u
        or v does not exist, they are added to the graph first.

        :param u: adjacent vertex
        :param v: adjacent vertex
        :return: nothing
        """
        # Return if nodes are the same.
        if u == v:
            return

        # Add nodes if they are not in list.
        if u not in self.adj_list:
            self.add_vertex(u)
        if v not in self.adj_list:
            self.add_vertex(v)

        # Check if edge already exists in the graph.
        if u in self.adj_list[v] or v in self.adj_list[u]:
            return

        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """Removes an between u and v edge from the graph. if u or v is not
        in the graph or no edge between u and v exists in the graph, nothing
        happens.

        :param v: adjacent vertex
        :param u: adjacent vertex
        :return: nothing
        """
        # Check if vertices and edge exists in the graph.
        if (
                v not in self.adj_list or u not in self.adj_list or
                v not in self.adj_list[u] or u not in self.adj_list[v]
        ):
            return

        # Delete edge from v
        u_index = self.adj_list[v].index(u)
        del self.adj_list[v][u_index]

        # Delete edge from u
        v_index = self.adj_list[u].index(v)
        del self.adj_list[u][v_index]

    def remove_vertex(self, v: str) -> None:
        """Removes a vertex and all incident edges from the graph. If the
        given vertex does not exist, nothing happens.

        :param v: vertex to delete
        :return: nothing
        """
        if v not in self.adj_list:
            return

        # Get first element in vertices list and remove the edge; continue
        # until vertices list is empty.
        while self.adj_list[v]:
            vertex = self.adj_list[v][0]
            self.remove_edge(v, vertex)

        # Delete the vertex.
        del self.adj_list[v]

    def get_vertices(self) -> []:
        """Return list of vertices in the graph (any order).

        :return: list graph vertices.
        """
        return [vertex for vertex in self.adj_list]

    def get_edges(self) -> []:
        """Return list of edges in the graph (any order). Each edge is
        represented by a tuple of incident vertices.

        :return: list of edges
        """
        edges = set()

        # Iterate over each vertex and add its incident vertices to the set
        # of edges.
        for head in self.adj_list:
            for tail in self.adj_list[head]:
                # Order the edge to avoid duplicates.
                edge = (head, tail)
                if tail >= head:
                    edge = (tail, head)

                # Duplicates are not added to sets.
                edges.add(edge)

        return list(edges)  # Order does not matter.

    def is_valid_path(self, path: []) -> bool:
        """Takes a list of vertex names and returns True if the sequence of
        vertices represents a valid path in the graph and False otherwise.
        Empty paths are considered valid.

        :param path: list of vertices
        :return: true or false
        """
        # One vertex in path, check if it's in the graph.
        if len(path) == 1:
            return path[0] in self.adj_list

        # Iterate over path vertices and check that the destination is in the
        # source's adjacency list.
        for v_index in range(len(path) - 1):
            src = path[v_index]
            dst = path[v_index + 1]
            if dst not in self.adj_list[src]:
                return False

        # Each vertex's subsequent destination was in its adjacency list.
        return True

    def dfs(self, v_start, v_end=None) -> []:
        """Return list of vertices visited during DFS search; vertices are
        picked in alphabetical order. If v_end is not provided it will search
        the entire graph. Otherwise, the search will stop when v_end is
        reached. If the starting vertex is not in the graph, nothing happens.

        :param v_start: search start vertex
        :param v_end: search end vertex or nothing
        :return: list of visited vertices
        """
        seen = []  # Use a list to preserve visited order.
        if v_start not in self.adj_list:
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
                # Push vertices to stack in ascending lexicographical order.
                for out_vertex in sorted(self.adj_list[vertex], reverse=True):
                    if out_vertex not in seen:
                        stack.append(out_vertex)

        return seen

    def bfs(self, v_start, v_end=None) -> []:
        """Return list of vertices visited during BFS search; vertices are
        picked in alphabetical order. If v_end is not provided it will search
        the entire graph. Otherwise, the search will stop when v_end is
        reached. If the starting vertex is not in the graph, nothing happens.

        :param v_start: search start vertex
        :param v_end: search end vertex
        :return: list of visited vertices
        """
        seen = []  # Use a list to preserve visited order.
        if v_start not in self.adj_list:
            # Start vertex not in graph.
            return seen

        queue = deque()
        queue.append(v_start)  # Enqueue start to the queue to begin search.
        while len(queue) > 0:
            vertex = queue.popleft()
            if vertex in seen:
                continue
            else:
                seen.append(vertex)
                if vertex == v_end:
                    # Target vertex found; end search.
                    break
                # Enqueue vertices to queue in ascending lexicographical order.
                for out_vertex in sorted(self.adj_list[vertex]):
                    if out_vertex not in seen:
                        queue.append(out_vertex)

        return seen

    def count_connected_components(self):
        """Returns the number of connected components in the graph. Returns zero
        for an empty graph.

        :return: number of connected components
        """
        connected_components = 0
        seen = set()
        for v_start in self.adj_list:
            if v_start not in seen:  # Skip visited vertices.
                # DFS will return all elements that are reachable from start.
                visited = self.dfs(v_start)
                seen.update(visited)  # Add visited elements to set.
                connected_components += 1

        return connected_components

    def _rec_has_cycle(self, vertex, prev, seen, cycle):
        """Uses recursion to perform a DFS so that it can keep track of the
        vertex that led to the current vertex. If current vertex has already
        been seen, cycle will be set to True. Otherwise, it remains False.

        :param vertex: current vertex
        :param prev: vertex that led to current vertex
        :param seen: set of visited vertices
        :param cycle: graph contains a cycle
        :return: true or false
        """
        if vertex in seen:
            # Vertex has already been seen; cycle exists.
            return True
        else:
            # Add visited vertex to seen.
            seen.add(vertex)

        for out_vertex in self.adj_list[vertex]:
            if out_vertex != prev:
                # Do not add previously visited vertex.
                cycle = self._rec_has_cycle(out_vertex, vertex, seen, cycle)

        # Return result of recursive call.
        return cycle

    def has_cycle(self):
        """Returns True if there is at least one cycle in the graph.

        :return: true or false
        """
        for v_start in self.adj_list:
            # Test each vertex for cycle using DFS.
            # Recursion makes it easier to keep track of the source vertex.
            if self._rec_has_cycle(v_start, None, set(), False):
                return True

        return False


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
