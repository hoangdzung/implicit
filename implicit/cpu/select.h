#ifndef IMPLICIT_CPU_SELECT_H_
#define IMPLICIT_CPU_SELECT_H_
#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include <iostream>

namespace implicit {

inline void select(const float * batch, int rows, int cols, int k, 
                   int * ids, float * distances) {
    std::vector<std::pair<float, int>> results;
    std::greater<std::pair<float, int> > heap_order;


    for (int row = 0; row < rows; ++row) {
        results.clear();
        for (int col = 0; col < cols; ++col) {
            float score = batch[row * cols + col];

	          if ((results.size() < k) || (score > results[0].first)) {
                if (results.size() >= k) {
                    std::pop_heap(results.begin(), results.end(), heap_order);
                    results.pop_back();
                }
                results.push_back(std::make_pair(score, col));
                std::push_heap(results.begin(), results.end(), heap_order);
            }
        }

		    std::sort_heap(results.begin(), results.end(), heap_order);

        for (size_t i = 0; i < results.size(); ++i) {
            ids[row * k + i] = results[i].second;
            distances[row * k + i] = results[i].first;
        }
    }
}
}
#endif  // IMPLICIT_TOPNC_H_
