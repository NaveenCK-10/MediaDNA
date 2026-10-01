# V22.2 Split Strategy Comparison

## A. Connected-Component Grouping (True Provenance)
- Retained: 21566
- All categories covered across all splits.
- Perfect component integrity and disjointness.

## B. Greedy Modularity Grouping (Previous)
- Retained: 18,701
- Dropped RVFA entirely from DEV/CAL/TEST due to false edges.

## Conclusion
By resolving the `fake.mp4` metadata linkage artifact, the graph naturally partitions into 10 perfectly isolated 50-identity components. Strategy A achieves 100% retention while guaranteeing mathematical disjointness.
