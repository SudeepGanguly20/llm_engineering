
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>

// Linear Congruential Generator
class LCG {
public:
    LCG(uint32_t seed) : value(seed) {}
    uint32_t next() {
        value = (a * value + c) % m;
        return value;
    }
private:
    uint32_t value;
    static constexpr uint32_t a = 1664525;
    static constexpr uint32_t c = 1013904223;
    static constexpr uint32_t m = (1ULL << 32);
};

// Brute-force max subarray sum
int max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    LCG lcg_gen(seed);
    std::vector<int> random_numbers(n);
    for (int i = 0; i < n; ++i) {
        random_numbers[i] = lcg_gen.next() % (max_val - min_val + 1) + min_val;
    }
    
    int max_sum = std::numeric_limits<int>::min();
    for (int i = 0; i < n; ++i) {
        int current_sum = 0;
        for (int j = i; j < n; ++j) {
            current_sum += random_numbers[j];
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }
    return max_sum;
}

// Function to calculate total max subarray sum over 20 runs
int total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    int total_sum = 0;
    LCG lcg_gen(initial_seed);
    for (int i = 0; i < 20; ++i) {
        uint32_t seed = lcg_gen.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    return total_sum;
}

int main() {
    int n = 10000;
    uint32_t initial_seed = 42;
    int min_val = -10;
    int max_val = 10;

    auto start = std::chrono::high_resolution_clock::now();
    int result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << '\n';
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}
