#include "XGESBatched.h"
#include "set_ops.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include <stack>
#include <algorithm>

using namespace std::chrono;

XGESBatched::XGESBatched(const int n_variables, ScorerInterface *scorer)
    : XGES(n_variables, scorer) {}

void XGESBatched::fit_xges(bool extended_search) {
    // Same as base class, but calls overridden heuristic_xges0
    XGES::fit_xges(extended_search);
}

void XGESBatched::heuristic_xges0(std::vector<Insert> &candidate_inserts,
                                  std::vector<Reverse> &candidate_reverses,
                                  std::vector<Delete> &candidate_deletes,
                                  UnblockedPathsMap &unblocked_paths_map,
                                  bool initialize_inserts) {
    if (initialize_inserts) {
        auto start_init_inserts = high_resolution_clock::now();
        
        std::vector<UnscoredInsert> unscored_inserts;
        // Pre-allocate to avoid reallocations
        unscored_inserts.reserve(n_variables * n_variables); 

        for (int y = 0; y < n_variables; ++y) {
            collect_inserts_to_y(y, unscored_inserts, -1);
        }
        
        process_inserts_batch(unscored_inserts, candidate_inserts, true);

        statistics["initialize_inserts-time"] += measure_time(start_init_inserts);
    }

    // The rest of the loop is identical to XGES::heuristic_xges0, 
    // but it calls our overridden update_operator_candidates_efficient
    
    EdgeModificationsMap edge_modifications;
    int i_operations = 1;

    Insert last_insert(-1, -1, FlatSet{}, -1, FlatSet{});

    while (!candidate_inserts.empty() || !candidate_reverses.empty() ||
           !candidate_deletes.empty()) {
        edge_modifications.clear();

        if (!candidate_deletes.empty()) {
            std::pop_heap(candidate_deletes.begin(), candidate_deletes.end());
            auto best_delete = std::move(candidate_deletes.back());
            candidate_deletes.pop_back();
            if (pdag.is_delete_valid(best_delete)) {
                pdag.apply_delete(best_delete, edge_modifications);
                total_score += best_delete.score;
                _logger->debug("{}: Delete applied", i_operations);
            } else {
                continue;
            }
        } else if (!candidate_reverses.empty()) {
            std::pop_heap(candidate_reverses.begin(), candidate_reverses.end());
            auto best_reverse = std::move(candidate_reverses.back());
            candidate_reverses.pop_back();
            if (pdag.is_reverse_valid(best_reverse, unblocked_paths_map)) {
                pdag.apply_reverse(best_reverse, edge_modifications);
                total_score += best_reverse.score;
                _logger->debug("{}: Reverse applied", i_operations);
            } else {
                continue;
            }
        } else if (!candidate_inserts.empty()) {
            std::pop_heap(candidate_inserts.begin(), candidate_inserts.end());
            auto best_insert = std::move(candidate_inserts.back());
            candidate_inserts.pop_back();
            if (best_insert.y == last_insert.y &&
                abs(best_insert.score - last_insert.score) < 1e-10 &&
                best_insert.x == last_insert.x && best_insert.T == last_insert.T) {
                statistics["probable_insert_duplicates"] += 1;
                continue;
            }
            last_insert = std::move(best_insert);
            if (pdag.is_insert_valid(last_insert, unblocked_paths_map)) {
                pdag.apply_insert(last_insert, edge_modifications);
                total_score += last_insert.score;
                _logger->debug("{}: Insert applied", i_operations);
            } else {
                continue;
            }
        }
        i_operations++;

        auto start_update = high_resolution_clock::now();
        update_operator_candidates_efficient(edge_modifications, candidate_inserts,
                                             candidate_reverses, candidate_deletes,
                                             unblocked_paths_map);
        auto end_update = high_resolution_clock::now();
        _logger->debug("Update candidates took {} ms", duration_cast<milliseconds>(end_update - start_update).count());
    }
}

void XGESBatched::update_operator_candidates_efficient(EdgeModificationsMap &edge_modifications,
                                                std::vector<Insert> &candidate_inserts,
                                                std::vector<Reverse> &candidate_reverses,
                                                std::vector<Delete> &candidate_deletes,
                                                UnblockedPathsMap &unblocked_paths_map) {
    // This logic is largely copied from XGES.cpp but adapted to use collect_* and process_*_batch
    
    auto start_time = high_resolution_clock::now();
    // First, undo all the edge modifications
    for (auto &[fst, edge_modification]: edge_modifications) {
        pdag.apply_edge_modification(edge_modification, true);
    }

    std::set<int> full_insert_to_y;
    std::map<int, std::set<int>> partial_insert_to_y;
    std::set<int> full_delete_to_y;
    std::set<int> full_delete_from_x;
    std::set<std::pair<int, int>> delete_x_y;
    std::set<int> full_reverse_to_y;
    std::set<int> full_reverse_from_x;
    std::set<std::pair<int, int>> reverse_x_y;

    // Re-apply the edge modifications one by one and update the operators
    // (Logic identical to XGES.cpp, omitted for brevity, assuming we can reuse the logic if we copy-paste or if we refactor XGES to expose this logic)
    // Since I cannot easily refactor XGES.cpp without editing it, I will duplicate the logic here.
    
    for (auto &[fst, edge_modification]: edge_modifications) {
        int a;
        int b;
        if (edge_modification.is_old_directed()) {
            a = edge_modification.get_old_source();
            b = edge_modification.get_old_target();
        } else if (edge_modification.is_new_directed()) {
            a = edge_modification.get_new_source();
            b = edge_modification.get_new_target();
        } else {
            a = edge_modification.x;
            b = edge_modification.y;
        }
        // Track inserts
        switch (edge_modification.get_modification_id()) {
            case 1:// a  b becomes a -- b
                full_insert_to_y.insert(a);
            case 2:// a  b becomes a → b
                full_insert_to_y.insert(b);
                std::ranges::set_intersection(
                        pdag.get_neighbors(a), pdag.get_neighbors(b),
                        std::inserter(full_insert_to_y, full_insert_to_y.begin()));
                for (auto target: pdag.get_neighbors(b)) {
                    partial_insert_to_y[target].insert(a);
                }
                for (auto target: pdag.get_neighbors(a)) {
                    partial_insert_to_y[target].insert(b);
                }
                break;
            case 3:// a -- b becomes a  b
                for (auto target: pdag.get_neighbors(b)) {
                    if (target == a) { continue; }
                    partial_insert_to_y[target].insert(a);
                }
                partial_insert_to_y[b].insert(a);
                partial_insert_to_y[a].insert(pdag.get_adjacent(b).begin(),
                                              pdag.get_adjacent(b).end());
                for (auto target: pdag.get_neighbors(a)) {
                    if (target == b) { continue; }
                    partial_insert_to_y[target].insert(b);
                }
                partial_insert_to_y[a].insert(b);
                partial_insert_to_y[b].insert(pdag.get_adjacent(a).begin(),
                                              pdag.get_adjacent(a).end());
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        partial_insert_to_y[y].insert(x);
                    }
                }
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        partial_insert_to_y[y].insert(x);
                    }
                }
                break;
            case 4:// a -- b becomes a → b
                partial_insert_to_y[a].insert(pdag.get_adjacent(b).begin(),
                                              pdag.get_adjacent(b).end());
                full_insert_to_y.insert(b);
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        partial_insert_to_y[y].insert(x);
                    }
                }
                break;
            case 5:// a → b becomes a  b
                for (auto target: pdag.get_neighbors(b)) {
                    partial_insert_to_y[target].insert(a);
                }
                partial_insert_to_y[b].insert(a);
                for (auto target: pdag.get_neighbors(a)) {
                    partial_insert_to_y[target].insert(b);
                }
                partial_insert_to_y[a].insert(b);
                full_insert_to_y.insert(b);
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        partial_insert_to_y[y].insert(x);
                    }
                }
                break;
            case 6:// a → b becomes a -- b
                full_insert_to_y.insert(a);
                full_insert_to_y.insert(b);
                break;
            case 7:// a → b becomes a ← b
                full_insert_to_y.insert(a);
                full_insert_to_y.insert(b);
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        partial_insert_to_y[y].insert(x);
                    }
                }
                break;
        }
        // Track deletes
        FlatSet x_intersection;
        switch (edge_modification.get_modification_id()) {
            case 1:// a  b becomes a -- b
                full_delete_to_y.insert(a);
            case 2:// a  b becomes a → b
                full_delete_to_y.insert(b);
                full_delete_from_x.insert(a);
                full_delete_from_x.insert(b);
                std::ranges::set_intersection(
                        pdag.get_adjacent(a), pdag.get_adjacent(b),
                        std::inserter(x_intersection, x_intersection.begin()));
                if (!x_intersection.empty()) {
                    FlatSet y_intersection;
                    std::ranges::set_intersection(
                            pdag.get_neighbors(a), pdag.get_neighbors(b),
                            std::inserter(y_intersection, y_intersection.begin()));
                    add_pairs(delete_x_y, x_intersection, y_intersection);
                }
                break;
            case 3:// a -- b becomes a  b
                break;
            case 4:// a -- b becomes a → b
            case 5:// a → b becomes a  b
                full_delete_to_y.insert(b);
                break;
            case 6:// a → b becomes a -- b
            case 7:// a → b becomes a ← b
                full_delete_to_y.insert(a);
                full_delete_to_y.insert(b);
                break;
        }

        // Track reverse (Simplified: we don't batch reverses yet as they are complex and fewer)
        // We will just use the original logic for reverses, which calls find_reverse_to_y_from_x
        // which calls scorer->score_reverse.
        // Ideally we should batch this too, but let's focus on Inserts and Deletes first as they are more frequent.
        switch (edge_modification.get_modification_id()) {
            case 1:// a  b becomes a -- b
                full_reverse_to_y.insert(a);
            case 2:// a  b becomes a → b
                full_reverse_to_y.insert(b);
                std::ranges::set_intersection(
                        pdag.get_neighbors(a), pdag.get_neighbors(b),
                        std::inserter(full_reverse_to_y, full_reverse_to_y.begin()));
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                break;
            case 3:// a -- b becomes a  b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({b, a});
                }
                break;
            case 4:// a -- b becomes a → b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(b);
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({b, a});
                }
                break;
            case 5:// a → b becomes a  b
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                break;
            case 6:// a → b becomes a -- b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(b);
                break;
            case 7:// a → b becomes a ← b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                break;
        }
        pdag.apply_edge_modification(edge_modification);
    }

    // --- BATCHED PROCESSING START ---
    std::vector<UnscoredInsert> unscored_inserts;
    std::vector<UnscoredDelete> unscored_deletes;

    // Find the inserts
    std::vector<int> keys_to_erase;
    for (const auto &[y, xs]: partial_insert_to_y) {
        if (full_insert_to_y.contains(y)) { keys_to_erase.push_back(y); }
    }
    for (auto key: keys_to_erase) { partial_insert_to_y.erase(key); }
    
    for (const auto &[y, xs]: partial_insert_to_y) {
        for (auto x: xs) {
            if (!pdag.get_adjacent(y).contains(x) && x != y) {
                collect_inserts_to_y(y, unscored_inserts, x);
            }
        }
    }
    for (auto y: full_insert_to_y) { collect_inserts_to_y(y, unscored_inserts); }

    // Find the deletes
    for (auto x: full_delete_from_x) { add_pairs(delete_x_y, x, pdag.get_neighbors(x)); }
    for (auto x: full_delete_from_x) { add_pairs(delete_x_y, x, pdag.get_children(x)); }
    for (auto y: full_delete_to_y) { add_pairs(delete_x_y, pdag.get_parents(y), y); }
    for (auto y: full_delete_to_y) { add_pairs(delete_x_y, pdag.get_neighbors(y), y); }
    for (auto [x, y]: delete_x_y) {
        if (x != y) { collect_delete_to_y_from_x(y, x, unscored_deletes); }
    }

    // Process batches
    process_inserts_batch(unscored_inserts, candidate_inserts, true);
    
    process_deletes_batch(unscored_deletes, candidate_deletes, true);

    // Find the reverses (Legacy non-batched for now)
    for (auto x: full_reverse_from_x) { add_pairs(reverse_x_y, x, pdag.get_parents(x)); }
    for (auto y: full_reverse_to_y) { add_pairs(reverse_x_y, pdag.get_children(y), y); }
    for (auto [x, y]: reverse_x_y) {
        if (pdag.has_directed_edge(y, x) && x != y) {
            find_reverse_to_y_from_x(y, x, candidate_reverses);
        }
    }
    statistics["update_operators-time"] += measure_time(start_time);
}

void XGESBatched::collect_inserts_to_y(int y, std::vector<UnscoredInsert> &unscored_inserts, int parent_x) const {
    auto &adjacent_y = pdag.get_adjacent(y);
    auto &parents_y = pdag.get_parents(y);

    std::set<int> possible_parents;

    if (parent_x != -1) {
        possible_parents.insert(parent_x);
    } else {
        auto &nodes = pdag.get_nodes_variables();
        std::ranges::set_difference(
                nodes, adjacent_y,
                std::inserter(possible_parents, possible_parents.begin()));
        possible_parents.erase(y);
    }

    for (int x: possible_parents) {
        auto neighbors_y_adjacent_x = pdag.get_neighbors_adjacent(y, x);
        if (!pdag.is_clique(neighbors_y_adjacent_x)) { continue; }

        auto neighbors_y_not_adjacent_x = pdag.get_neighbors_not_adjacent(y, x);
        FlatSet effective_parents_y = neighbors_y_adjacent_x;
        effective_parents_y.insert(parents_y.begin(), parents_y.end());
        
        std::stack<std::tuple<FlatSet, FlatSet::iterator, FlatSet>> stack;
        stack.emplace(FlatSet{}, neighbors_y_not_adjacent_x.begin(), effective_parents_y);

        while (!stack.empty()) {
            auto top = std::move(stack.top());
            stack.pop();
            auto &T = std::get<0>(top);
            auto it = std::get<1>(top);
            auto &effective_parents = std::get<2>(top);

            // Instead of scoring, collect
            unscored_inserts.push_back({x, y, T, effective_parents});

            while (it != neighbors_y_not_adjacent_x.end()) {
                auto z = *it;
                ++it;
                auto &adjacent_z = pdag.get_adjacent(z);
                if (std::ranges::includes(adjacent_z, T) &&
                    std::ranges::includes(adjacent_z, neighbors_y_adjacent_x)) {
                    auto T_prime = T;
                    T_prime.insert(z);
                    auto effective_parents_prime = effective_parents;
                    effective_parents_prime.insert(z);
                    stack.emplace(std::move(T_prime), it,
                                  std::move(effective_parents_prime));
                }
            }
        }
    }
}

void XGESBatched::collect_delete_to_y_from_x(int y, int x, std::vector<UnscoredDelete> &unscored_deletes) const {
    const FlatSet &parents_y = pdag.get_parents(y);
    auto neighbors_y_adjacent_x = pdag.get_neighbors_adjacent(y, x);
    bool directed_xy = pdag.has_directed_edge(x, y);

    std::stack<std::tuple<FlatSet, FlatSet::iterator, FlatSet>> stack;
    FlatSet effective_parents_init;
    effective_parents_init.reserve(parents_y.size() + neighbors_y_adjacent_x.size() + 1);
    union_with_single_element(parents_y, x, effective_parents_init);
    stack.emplace(FlatSet{}, neighbors_y_adjacent_x.begin(), effective_parents_init);

    while (!stack.empty()) {
        auto top = std::move(stack.top());
        stack.pop();
        auto C = std::get<0>(top);
        auto it = std::get<1>(top);
        auto effective_parents = std::get<2>(top);

        // Instead of scoring, collect
        unscored_deletes.push_back({x, y, C, effective_parents, directed_xy});

        while (it != neighbors_y_adjacent_x.end()) {
            auto z = *it;
            ++it;
            auto &adjacent_z = pdag.get_adjacent(z);
            if (std::ranges::includes(adjacent_z, C)) {
                auto C_prime = C;
                C_prime.insert(z);
                auto effective_parents_prime = effective_parents;
                effective_parents_prime.erase(z);
                stack.emplace(std::move(C_prime), it,
                              std::move(effective_parents_prime));
            }
        }
    }
}

void XGESBatched::process_inserts_batch(const std::vector<UnscoredInsert>& unscored, std::vector<Insert>& candidates, bool positive_only) {
    if (unscored.empty()) return;

    std::vector<int> targets;
    std::vector<FlatSet> parents_list;
    targets.reserve(unscored.size() * 2);
    parents_list.reserve(unscored.size() * 2);

    for(const auto& u : unscored) {
        // Score with new parent (add x)
        targets.push_back(u.y);
        FlatSet parents_with_x = u.effective_parents;
        parents_with_x.insert(u.x);
        parents_list.push_back(parents_with_x);
        
        // Score without new parent (x is not in effective_parents)
        targets.push_back(u.y);
        parents_list.push_back(u.effective_parents);
    }

    std::vector<double> scores;
    scorer->local_score_batched(scores, targets, parents_list);

    if (!scores.empty()) {
        _logger->info("Batch scores example: with={} without={} diff={}", scores[0], scores[1], scores[0]-scores[1]);
    }

    for(size_t i=0; i<unscored.size(); ++i) {
        double score_with = scores[2*i];
        double score_without = scores[2*i+1];
        double score_diff = score_with - score_without;

        if (score_diff > 0 || !positive_only) {
            candidates.emplace_back(unscored[i].x, unscored[i].y, unscored[i].T, score_diff, unscored[i].effective_parents);
            std::push_heap(candidates.begin(), candidates.end());
        }
    }
}

void XGESBatched::process_deletes_batch(const std::vector<UnscoredDelete>& unscored, std::vector<Delete>& candidates, bool positive_only) {
    if (unscored.empty()) return;

    std::vector<int> targets;
    std::vector<FlatSet> parents_list;
    targets.reserve(unscored.size() * 2);
    parents_list.reserve(unscored.size() * 2);

    for(const auto& u : unscored) {
        // Score without old parent (remove x)
        targets.push_back(u.y);
        FlatSet parents_without_x = u.effective_parents;
        parents_without_x.erase(u.x);
        parents_list.push_back(parents_without_x);

        // Score with old parent (x is already in effective_parents)
        targets.push_back(u.y);
        parents_list.push_back(u.effective_parents);
    }

    std::vector<double> scores;
    scorer->local_score_batched(scores, targets, parents_list);

    for(size_t i=0; i<unscored.size(); ++i) {
        double score_without = scores[2*i];
        double score_with = scores[2*i+1];
        double score_diff = score_without - score_with;

        if (score_diff > 0 || !positive_only) {
            candidates.emplace_back(unscored[i].x, unscored[i].y, unscored[i].C, score_diff, unscored[i].effective_parents, unscored[i].directed_xy);
            std::push_heap(candidates.begin(), candidates.end());
        }
    }
}
