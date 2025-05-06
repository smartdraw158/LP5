#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void parallelBFS(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    cout << "Parallel BFS Traversal: ";

    while (!q.empty()) {
        int qSize;

        #pragma omp critical
        {
            qSize = q.size();
        }

        vector<int> currentLevel(qSize);
        
        // Extract current level nodes in a thread-safe way
        for (int i = 0; i < qSize; i++) {
            #pragma omp critical
            {
                currentLevel[i] = q.front();
                q.pop();
            }
        }

        vector<int> nextLevel;

        // Process current level nodes in parallel
        #pragma omp parallel for shared(nextLevel)
        for (int i = 0; i < qSize; i++) {
            int node = currentLevel[i];

            #pragma omp critical
            {
                cout << node << " ";
            }

            for (int j = 0; j < graph[node].size(); j++) {
                int adj = graph[node][j];

                bool needVisit = false;
                #pragma omp critical
                {
                    if (!visited[adj]) {
                        visited[adj] = true;
                        needVisit = true;
                    }
                }

                if (needVisit) {
                    #pragma omp critical
                    {
                        nextLevel.push_back(adj);
                    }
                }
            }
        }

        // Push next level nodes to queue
        for (int i = 0; i < nextLevel.size(); i++) {
            q.push(nextLevel[i]);
        }
    }

    cout << endl;
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

    // Call parallel BFS
    parallelBFS(start_node);

    return 0;
}

