# Box contacts with retained convex hulls

`segment_hull::PreparedSegmentHull::box_contact` returns up to four contact witnesses
for an oriented box against the retained, closed convex hull. Positions, extents,
axes and returned witnesses are f64 in the hull's local frame. The input box frame
must be right-handed and orthonormal; half extents must be positive and the prediction
margin nonnegative. Invalid inputs and nonfinite computations return `BoxHullError`.

The normal points from hull to box. Negative separation denotes penetration;
positive separation denotes a predicted contact admitted by the margin. Each point
has a box witness, hull witness and signed separation. Face normals, box normals
and supporting edge-cross axes all participate in SAT. Incident faces are clipped
to the opposing finite face, not its infinite plane or bounding box. Edge contacts
use closest segment witnesses. At most four points are retained in polygon order.

The existing prepared hull now retains the two opposite triangle vertices of each
edge. That supplies a local outward-normal-cone test, so edge axes do not repeatedly
scan every hull vertex. Preparation still sorts triangulation edges in O(E log E),
retaining O(V + F + E) geometry. SAT work is O(F + E + V) for one fixed-topology box;
face clipping can visit O(F) planes and grow with the clipped polygon's size. Per-call
work counters report axes, vertex projections and clipping planes. This first path
allocates temporary clipping polygons; it makes no allocation-free claim.

This is a pose/contact kernel, not a translation or rotational sweep. The consuming
physics engine owns speculative admission, velocity/angular bounds, response,
persistence, wake/sleep and CCD. An adapter must not claim general swept box/hull or
hull/hull support from this operation alone.

Tests compare 512 deterministic rotated box/cube cases with the separate primitive
box SAT query, check signed face depths and finite-shape witnesses, and preserve
the existing segment-distance tests. The concrete consumer is Physics Engine's
Greyhaven crate/panel Rapier oracle, not a new game-side solver.

For repeated translation queries, `PreparedBoxHull::new(Arc<PreparedSegmentHull>, half,
axes, work)` retains the hull and the box-frame SAT planes. `contact(centre, margin,
work)` visits those planes without repeating hull projections or edge-cone construction.
A new frame is required when orientation, extents or geometry change. Owning an `Arc`
prevents applying a frame to a different hull; translation remains per-call input.
`retained_bytes` counts the frame's owned plane payload, excluding shared hull geometry.
Preparation work and warm query work have separate counters. Tests compare warm-frame
queries bit-for-bit with fresh preparation at different translations and margins.

Face normalization and edge-cone normals are retained with the hull. Near a cone boundary,
the query retains the direct triple-product classification. Separated finite-face witnesses
that attain the supporting-plane gap may finish without the remaining SAT axes; edges,
corners and penetration retain the full search. Contact polygons still use temporary
allocation; this does not establish an allocation-free query or rotational sweep.
