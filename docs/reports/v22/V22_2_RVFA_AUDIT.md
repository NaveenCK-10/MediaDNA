# V22.2 RVFA Category Audit

## Root Cause Analysis
In the previous split generation, RVFA samples were completely dropped from DEV, CAL, and TEST. This was tracked down to a metadata parsing artifact. RVFA samples use `fake.mp4` as the `target1` value to indicate the audio was synthetic. The previous graph builder treated `fake.mp4` as a valid identity node, which created massive artificial cross-edges linking hundreds of unrelated identities into a single giant component.

The greedy modularity algorithm clustered these together and assigned the largest community to TRAIN. By ignoring `fake.mp4` and other non-identity placeholders during edge construction, the graph naturally separates into exactly 10 independent components of 50 identities each. The RVFA samples are perfectly distributed among these 10 components.

## Current Counts
- Original Dataset RVFA: 500
- TRAIN RVFA: 350
- DEV RVFA: 50
- CAL RVFA: 50
- TEST RVFA: 50
