#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <queue>
#include <limits>

struct Edge {
    size_t u, v;
    int64_t weight;
};

class SteinerTreeSolver {
    struct EdgeTo {
        size_t to;
        int64_t weight;
    };

    using AdjList = std::vector<std::vector<EdgeTo>>;

public:
    SteinerTreeSolver(size_t n_nodes, std::vector<Edge>&& edges, std::vector<size_t>&& terminals)
        : n_nodes_(n_nodes)
        , edges_(std::move(edges))
        , terminals_(std::move(terminals))
    {}

    void Run() {
        // build adjacency list for dijkstra
        adj_list_.resize(n_nodes_);
        for (Edge& edge : edges_) {
            adj_list_[edge.u].push_back({edge.v, edge.weight});
            adj_list_[edge.v].push_back({edge.u, edge.weight});
        }

        AdjList terminal_closure = GetTerminalMetricClosure();
        std::vector<Edge> terminal_mst = GetMST(terminal_closure);
        std::set<size_t> used_nodes;

        for (const Edge& mst_edge : terminal_mst) {
            size_t u = terminals_[mst_edge.u];
            size_t v = terminals_[mst_edge.v];
            std::vector<size_t> path;
            GetMinPath(u, v, &path);
            // path \cap current solution
            std::vector<size_t> node_intersection;
            for (size_t w : path) {
                if (used_nodes.count(w)) {
                    node_intersection.push_back(w);
                }
            }
            if (node_intersection.size() < 2) {
                for (size_t i = 0; i < path.size() - 1; ++i) {
                    solution_.push_back({path[i], path[i + 1]});
                }
            } else {
                AddSubpath(u, node_intersection.front(), path);
                AddSubpath(node_intersection.back(), v, path);
            }
        }
    }

    std::vector<std::pair<size_t, size_t>> GetSolution() {
        return solution_;
    }

private:
    class DisjointSetUnion {
    public:
        DisjointSetUnion(size_t n) {
            root_.resize(n);
            for (size_t i = 0; i < n; ++i) {
                root_[i] = i;
            }
            size_.resize(n, 1);
        }

        size_t GetRoot(size_t u) {
            return u == root_[u] ? u : (root_[u] = GetRoot(root_[u]));
        }

        void Unite(size_t u, size_t v) {
            u = GetRoot(u);
            v = GetRoot(v);
            if (u == v) {
                return;
            }
            if (size_[u] < size_[v]) {
                std::swap(u, v);
            }
            size_[u] += size_[v];
            root_[v] = u;
        }

    private:
        std::vector<size_t> root_;
        std::vector<size_t> size_;
    };

    AdjList GetTerminalMetricClosure() {
        AdjList terminal_closure(terminals_.size());
        for (size_t i = 0; i < terminals_.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                int64_t min_dist = GetMinPath(terminals_[i], terminals_[j]);
                terminal_closure[i].push_back({j, min_dist});
                terminal_closure[j].push_back({i, min_dist});
            }
        }
        return terminal_closure;
    }

    // Kruskal
    static std::vector<Edge> GetMST(const AdjList& adj_list) {
        std::vector<Edge> edges;
        for (size_t u = 0; u < adj_list.size(); ++u) {
            for (const EdgeTo& edge : adj_list[u]) {
                edges.push_back({u, edge.to, edge.weight});
            }
        }

        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.weight < b.weight;
        });

        std::vector<Edge> mst;
        auto dsu = DisjointSetUnion(adj_list.size());
        for (const Edge& edge : edges) {
            if (dsu.GetRoot(edge.u) != dsu.GetRoot(edge.v)) {
                mst.push_back(edge);
                dsu.Unite(edge.u, edge.v);
            }
        }

        return mst;
    }

    // dijkstra
    // |terminals| * O(dijkstra) <= O(floyd), if there are not many terminal nodes
    int64_t GetMinPath(size_t start, size_t finish, std::vector<size_t>* path = nullptr) {
        std::vector<int64_t> dist(n_nodes_, std::numeric_limits<int64_t>::max());
        dist[start] = 0;
        std::priority_queue<std::pair<int64_t, size_t>> queue;
        queue.push({0, start});

        std::vector<size_t> prev_node(n_nodes_ * (path != nullptr), -1);

        while (!queue.empty()) {
            size_t u = queue.top().second;
            if (-queue.top().first > dist[u]) {
                queue.pop();
                continue;
            }
            queue.pop();
            for (const EdgeTo& edge : adj_list_[u]) {
                if (dist[edge.to] > dist[u] + edge.weight) {
                    dist[edge.to] = dist[u] + edge.weight;
                    queue.push({-dist[edge.to], edge.to});
                    if (path != nullptr) {
                        prev_node[edge.to] = u;
                    }
                }
            }
        }

        if (path != nullptr) {
            // trace back
            path->push_back(finish);
            for (size_t u = finish; u != start; ) {
                path->push_back(prev_node[u]);
                u = prev_node[u];
            }
            std::reverse(path->begin(), path->end());
        }

        return dist[finish];
    }

    void AddSubpath(size_t u, size_t v, const std::vector<size_t>& path) {
        size_t begin = 0;
        while (path[begin] != u) ++begin;
        size_t end = 0;
        while (path[end] != v) ++end;
        for (size_t i = begin; i < end; ++i) {
            solution_.push_back({path[i], path[i + 1]});
        }
    }

    size_t n_nodes_;
    std::vector<Edge> edges_;
    std::vector<size_t> terminals_;
    AdjList adj_list_;
    std::vector<std::pair<size_t, size_t>> solution_;
};

int main() {
    freopen("input.txt", "r", stdin);
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);

    size_t n_nodes;
    size_t n_edges;
    size_t n_terminals;

    std::vector<Edge> edges;
    std::vector<size_t> terminals;

    std::string tmp;
    do {
        std::cin >> tmp;
    } while (tmp != "Nodes");

    std::cin >> n_nodes >> tmp >> n_edges;
    for (size_t i = 0; i < n_edges; ++i) {
        size_t u, v;
        int64_t weight;
        std::cin >> tmp >> u >> v >> weight;
        edges.push_back({u - 1, v - 1, weight});
    }

    do {
        std::cin >> tmp;
    } while (tmp != "Terminals");

    std::cin >> tmp >> n_terminals;
    for (size_t i = 0; i < n_terminals; ++i) {
        size_t u;
        std::cin >> tmp >> u;
        terminals.push_back(u - 1);
    }

    auto solver = SteinerTreeSolver(n_nodes, std::move(edges), std::move(terminals));
    solver.Run();
    for (const auto& edge : solver.GetSolution()) {
        std::cout << edge.first + 1 << " " << edge.second + 1 << std::endl;
    }
}
