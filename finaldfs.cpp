#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void dfsTask(int node) {
    #pragma omp critical
    {
        if (visited[node]) return;
        visited[node] = true;
        cout << node << " ";
    }

    for (int i = 0; i < graph[node].size(); i++) {
        int adj = graph[node][i];
        #pragma omp task
        dfsTask(adj);
    }
}

void parallelDFS(int start_node) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            dfsTask(start_node);
        }
    }
}

int main() {
    int n, m, start_node;
    cout << "Enter No of Nodes, Edges, and Start Node: ";
    cin >> n >> m >> start_node;

    cout << "Enter Pairs of Edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // Undirected graph
    }

    // Initialize visited array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    cout << "Parallel DFS Traversal: ";
    parallelDFS(start_node);
    cout << endl;

    return 0;
}

