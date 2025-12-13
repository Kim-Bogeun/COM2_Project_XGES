#pragma once

#include "XGES.h"
#include <vector>
#include <tuple>

class XGESBatched : public XGES {
public:
    XGESBatched(const int n_variables, ScorerInterface *scorer);

    void fit_xges(bool extended_search) override;

protected:
    // Override the heuristic to use batching
    void heuristic_xges0(std::vector<Insert> &candidate_inserts,
                         std::vector<Reverse> &candidate_reverses,
                         std::vector<Delete> &candidate_deletes,
                         UnblockedPathsMap &unblocked_paths_map,
                         bool initialize_inserts) override;

    // Override the update function to use batching
    void update_operator_candidates_efficient(EdgeModificationsMap &edge_modifications,
                                              std::vector<Insert> &candidate_inserts,
                                              std::vector<Reverse> &candidate_reverses,
                                              std::vector<Delete> &candidate_deletes,
                                              UnblockedPathsMap &unblocked_paths_map) override;

private:
    // Helper structures for batching
    struct UnscoredInsert {
        int x;
        int y;
        FlatSet T;
        FlatSet effective_parents;
    };

    struct UnscoredDelete {
        int x;
        int y;
        FlatSet C;
        FlatSet effective_parents;
        bool directed_xy;
    };

    struct UnscoredReverse {
        int x;
        int y;
        FlatSet C;
        FlatSet effective_parents_y; // Parents of y after reverse (without x)
        FlatSet effective_parents_x; // Parents of x after reverse (with y)
        bool directed_xy;
    };

    // Batched versions of find functions
    void collect_inserts_to_y(int y, std::vector<UnscoredInsert> &unscored_inserts, int parent_x = -1) const;
    void collect_delete_to_y_from_x(int y, int x, std::vector<UnscoredDelete> &unscored_deletes) const;
    // void collect_reverse_to_y_from_x(int y, int x, std::vector<UnscoredReverse> &unscored_reverses) const; // Not implemented for now as reverse is complex

    // Process the collected unscored candidates
    void process_inserts_batch(const std::vector<UnscoredInsert>& unscored, std::vector<Insert>& candidates, bool positive_only);
    void process_deletes_batch(const std::vector<UnscoredDelete>& unscored, std::vector<Delete>& candidates, bool positive_only);
};
