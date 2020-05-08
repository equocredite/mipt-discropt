#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>

/*double ElapsedTime() {
    static clock_t start = clock();
    return static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
}*/

struct Item {
    int64_t volume;
    int64_t value;
};


class KnapsackSolver {
    using iterator_t = std::vector<Item>::iterator;

public:
    KnapsackSolver(int64_t knapsack_volume, const std::vector<Item>& items)
        : knapsack_volume_(knapsack_volume)
        , items_(items)
        , lower_bound_(0)
    {}

    void Run() {
        std::sort(items_.begin(), items_.end(), [](const Item& a, const Item& b) {
            return static_cast<double>(a.value) / a.volume > static_cast<double>(b.value) / b.volume;
        });
        lower_bound_ = GetLowerBoundOnSuffix(items_.begin(), knapsack_volume_);
        BruteForce(items_.begin(), knapsack_volume_, 0);
    }

    int64_t GetAnswer() {
        return lower_bound_;
    }

private:
    void BruteForce(iterator_t it, int64_t volume, int64_t value) {
        if (it == items_.end()) {
            UpdateAnswer(value);
            return;
        }

        std::vector<int64_t> take_order;
        if (rand() % 2) take_order = {1, 0};
        else take_order = {0, 1};

        for (int64_t take : take_order) {
            int64_t new_volume = volume - it->volume * take;
            if (new_volume < 0) continue;
            int64_t new_value = value + it->value * take;
            int64_t upper_bound = new_value + GetUpperBoundOnSuffix(it + 1, new_volume);
            if (upper_bound >= lower_bound_) {
                UpdateAnswer(new_value + GetLowerBoundOnSuffix(it + 1, new_volume));
                BruteForce(it + 1, new_volume, new_value);
            }
        }
    }

    void UpdateAnswer(int64_t value) {
        lower_bound_ = std::max(lower_bound_, value);
    }

    int64_t GetUpperBoundOnSuffix(iterator_t suffix, int64_t volume_int) {
        double volume = static_cast<double>(volume_int);
        return static_cast<int64_t>(std::accumulate(suffix, items_.end(), 0., [&volume](double acc, const Item& item) {
            if (volume > item.volume) {
                volume -= item.volume;
                return acc + item.value;
            }
            double fitting_fraction = volume / item.volume;
            volume = 0.;
            return acc + fitting_fraction * item.value;
        }));
    }

    int64_t GetLowerBoundOnSuffix(iterator_t suffix, int64_t volume) {
        return std::accumulate(suffix, items_.end(), 0, [&volume](int64_t acc, const Item& item) {
            if (item.volume > volume) {
                return acc;
            }
            volume -= item.volume;
            return acc + item.value;
        });
    }

    int64_t  knapsack_volume_;
    std::vector<Item> items_;
    // maximal answer that we already can guarantee
    int64_t lower_bound_;
};


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    srand(time(0));

    int64_t knapsack_volume;
    size_t items_cnt;
    std::cin >> knapsack_volume >> items_cnt;
    std::vector<Item> items(items_cnt);
    for (Item& item : items) {
        std::cin >> item.volume >> item.value;
    }

    KnapsackSolver solver(knapsack_volume, items);
    solver.Run();
    std::cout << solver.GetAnswer() << std::endl;
}
