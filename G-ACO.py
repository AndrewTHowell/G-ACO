import numpy as np


class Ant:
    def __init__(self, graph, vertices, params):
        self.dGraph = graph
        self.vertices = vertices
        self.startPosition = np.random.choice(self.vertices)
        self.alpha, self.beta = params

    def buildTour(self, pGraph) -> (list, np.array):
        dGraph = self.dGraph
        vertices = self.vertices
        currentVertex = self.startPosition

        tour = []
        traversedEdges = np.zeros_like(self.dGraph)
        while len(dGraph) > 0:
            nextVertex = self.chooseVertex(currentVertex, vertices, pGraph[currentVertex], dGraph[currentVertex])

            tour.append(nextVertex)
            traversedEdges[currentVertex][nextVertex] = 1
            traversedEdges[nextVertex][currentVertex] = 1

            vertices.remove(currentVertex)
            pGraph = np.delete(np.delete(pGraph, currentVertex, 0), currentVertex, 1)
            dGraph = np.delete(np.delete(dGraph, currentVertex, 0), currentVertex, 1)

            currentVertex = nextVertex

        # Head to start position for valid tour
        tour.append(self.startPosition)
        traversedEdges[currentVertex][self.startPosition] = 1
        traversedEdges[self.startPosition][currentVertex] = 1

        return tour, traversedEdges

    def chooseVertex(self, currentVertex, vertices, pGraph, dGraph) -> int:
        if len(vertices) == 1:
            return vertices[0]
        heuristics = (pGraph ** self.alpha) * (dGraph ** self.beta)
        normalisedHeuristics = heuristics/heuristics.sum()

        chosenVertex = np.random.choice(vertices, p=normalisedHeuristics)

        return chosenVertex


class GeneralAntColonyOptimisation:
    def __init__(self, graph, maxIterations, numAnts, p0, pRate, antParams):
        self.dGraph = np.array(graph)
        self.vertices = [vertex for vertex in range(len(self.dGraph))]

        self.maxIterations = maxIterations
        self.numAnts = numAnts
        self.antParams = antParams

        self.p0 = p0
        self.pRate = pRate

    def run(self) -> list:
        # p0 pheromones
        pGraph = np.zeros_like(self.dGraph) + self.p0

        # bestTour defaults to ordered vertices plus the original vertex to create a tour
        bestTour = self.vertices + [1]
        bestTourCost = self.tourCost(bestTour)

        # Initialise ants
        ants = []
        for _ in range(self.numAnts):
            # Initialise an ant at a vertex
            ants.append(Ant(self.dGraph, self.vertices, self.antParams))

        while (t := 0) < self.maxIterations:
            totalEdgesTraversed = np.zeros_like(self.dGraph)
            for ant in ants:
                tour, edgesTraversed = ant.buildTour(pGraph)

                tourCost = self.tourCost(tour)

                # Pheromones contributed by this ant are scaled so shorter tours leave more pheromones
                totalEdgesTraversed += edgesTraversed * (1 / tourCost)

                if tourCost < bestTourCost:
                    bestTour, bestTourCost = tour, tourCost

            # Evaporate pheromone levels
            pGraph *= 1 - self.pRate

            # Deposit pheromones
            pGraph += totalEdgesTraversed

            t += 1

        return bestTour

    def tourCost(self, tour) -> int:
        cost: int = 0
        for edge in zip(tour, tour[1:]):
            cost += self.dGraph[edge[0]][edge[1]]
        return cost


if __name__ == "__main__":
    G = [[0, 1, 3],
         [1, 0, 2],
         [3, 2, 0]]

    solver = GeneralAntColonyOptimisation(G, 100, 30, 0.01, 0.8, (1,1))

    finalTour = solver.run()

    print(finalTour)
