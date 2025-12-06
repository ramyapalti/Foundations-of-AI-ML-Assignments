import sys
from graph import Graph
import time


def shortest_path(graph, start_vertex, end_vertex):
    # Given the graph, find the shortest path between the start and end vertex
    # return path as list of vertices
    # path = [start_vertex, node1, node2, ..., end_vertex]

    # graph has vertices, and edges (dictionary of lists)
    # vertices[i] = [x, y] coordinates of vertex i
    # edges[i] = list of nodes j that are adjacent to node i
    
    return []
    
# printing path, time taken by the shortest path algorithm and length of the path
def print_shortest_path(graph, start_vertex, end_vertex):
    start_time = time.time()
    path = shortest_path(graph, start_vertex, end_vertex)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Time taken by the program: {:.2f} seconds".format(elapsed_time))
    print("Length of the path: ", len(path))
    print("Path:\n", path)
    return path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python shortest_path.py <graph_file>")
        sys.exit(1)

    graph_file = sys.argv[1]
    n = int(graph_file.split("_")[1].split(".")[0]) # side length of square grid
    # load the graph with the graph file
    graph = Graph(n)
    graph.load("./data/" + graph_file)

    start_vertex, end_vertex = 0, len(graph.vertices) -1

    path = print_shortest_path(graph, start_vertex, end_vertex)

    # plot path and graph 
    graph.plot(path=path)

    # check if the path exists in the graph
    is_shortest = graph.path_checker(path)

    if is_shortest:
        print("\nThe given path exists in the graph")
    else:
        print("\nThe given path does NOT exist in the graph")