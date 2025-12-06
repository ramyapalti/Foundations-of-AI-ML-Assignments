import random
from matplotlib import pyplot as plt

class Graph:
    def __init__(self, n):
        self.n = n
        self.vertices = {}
        self.edges = {}

    def generate(self, n, p):
        # generates each edge with probability p
        # at most 4 edges per vertex
        for i in range(n):
            for j in range(n):
                vertex = i * n + j
                self.vertices[vertex] = [i, j]
                self.edges[vertex] = []

                if i > 0 and random.random() < p:
                    self.add_edge(vertex, vertex - n)
                if j > 0 and random.random() < p:
                    self.add_edge(vertex, vertex - 1)

    #adds an edge between u and v
    def add_edge(self, u, v):
        if u not in self.edges:
            self.edges[u] = []
        if v not in self.edges:
            self.edges[v] = []
        self.edges[u].append(v)
        self.edges[v].append(u)

    # saves the graph to a file
    def save(self, filename):
        n = int(len(self.vertices) ** 0.5)
        with open(filename, 'w') as f:
            f.write(str(n) + '\n')
            for vertex, coords in self.vertices.items():
                row, col = coords
                f.write(f"{vertex} ({row}, {col})\n")
            for vertex, neighbors in self.edges.items():
                f.write(f"{vertex}")
                for neighbor in neighbors:
                    f.write(f" {neighbor}")
                f.write('\n')

    # loads the graph from a file
    def load(self, filename):
        with open(filename, 'r') as f:
            n = int(f.readline()) # first line is the square side length
            self.n = n

            # node_number (x, y)
            for _ in range(n * n):
                line = f.readline().split(" ", 1)
                vertex = int(line[0])
                coords = line[1].strip('()\n').split(',')
                row, col = map(int, coords)
                self.vertices[vertex] = [row, col]
                self.edges[vertex] = []

            # node_number neighbor1 neighbor2 ...
            for line in f:
                vertices = line.split()
                u = int(vertices[0])
                neighbors = list(map(int, vertices[1:]))
                for v in neighbors:
                    self.add_edge(u, v)

    # plots the graph, path is optional
    # if path is given, it will be plotted in red
    def plot(self, path=None):
        fig, ax = plt.subplots(figsize=(self.n , self.n))
        for u, neighbors in self.edges.items():
            for v in neighbors:
                x1, y1 = self.vertices[u]
                x2, y2 = self.vertices[v]
                ax.plot([x1, x2], [y1, y2], 'k-')
        ax.plot([v[1] for v in self.vertices.values()], [v[0] for v in self.vertices.values()], 'bo')
        
        if path:
            path_x = [self.vertices[v][0] for v in path]
            path_y = [self.vertices[v][1] for v in path]
            ax.plot(path_x, path_y, 'r-')
        
            plt.title('Planar graph with %i nodes' % len(self.vertices))
            plt.savefig(f"./results/path_{self.n}.png")
        
        else:
            plt.title('Planar graph with %i nodes' % len(self.vertices))
            plt.savefig(f"./data/graph_{self.n}.png")

    # checks if the graph is connected
    def is_connected(self):
        """
        Returns True if the graph is connected, False otherwise.
        """
        if not self.vertices:
            # An empty graph is not connected by definition
            return False

        # Initialize the set of visited vertices with the first vertex
        visited = {list(self.vertices.keys())[0]}

        # Keep exploring the graph until all vertices have been visited
        while len(visited) < len(self.vertices):
            # Find an unvisited vertex that is adjacent to a visited vertex
            found = False
            for u in visited:
                for v in self.edges[u]:
                    if v not in visited:
                        visited.add(v)
                        found = True
                        break
                if found:
                    break

            # If we couldn't find an unvisited adjacent vertex, the graph is not connected
            if not found:
                return False

        return True
    
    # checks if the path is valid
    def path_checker(self, path):
        # Check if the path is empty or has only one vertex
        if not path or len(path) == 1:
            return False

        # Check if each consecutive pair of vertices in the path has an edge between them
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if v not in self.edges.get(u, []):
                return False

        return True


if __name__ == "__main__":
    
    for i in range(10, 110, 10):
        print(f"Creating graph with {i} nodes")
        while 1:
            g = Graph(i)
            g.generate(i, 0.9)
            if g.is_connected():
                print("The graph is connected")
                g.save(f"./data/graph_{i}.txt")
                g.plot()
                break
            else:
                print("The graph is not connected")
                continue
        print()