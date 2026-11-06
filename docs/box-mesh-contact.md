# Box contacts with retained triangle meshes

`PreparedTriangleMesh3::box_contacts_counted` queries a fixed oriented box against the
existing immutable, two-sided mesh BVH. Centre, positive half extents, right-handed
orthonormal axes and a nonnegative prediction margin are f64 in mesh-local coordinates.
Invalid input or nonfinite computation returns `TriangleMeshError3`.

The callback receives a stable input triangle ID, a normal from triangle to box, and up
to four finite-face or edge witnesses. Each witness has `point_box`, `point_mesh` and a
signed separation. Duplicate boundary witnesses are removed before selecting four points,
so shared-edge intersections cannot displace a real supporting corner. SAT tests the triangle normal, box normals and edge-cross axes.
Incident faces are clipped to the finite opposing feature. A separated face witness
attaining its supporting-plane gap can certify a result before the remaining axes.
Open boundaries and holes remain open; a triangle's bounding box is never its collider.

The query traverses the retained BVH with the box bounds expanded by the prediction
margin. Work counters include visited nodes and exact triangle tests. The query does
not rebuild mesh geometry. Clipping uses temporary vectors and makes no allocation-free
claim. Callback order follows deterministic BVH traversal; triangle identity remains the
input index. An error may follow earlier emissions, so callers must discard the whole
query's emissions on error.

This is pose/contact geometry, not swept time of impact. Physics Engine owns interval
prediction, response, persistence, wake/sleep and explicit rejection of unsupported CCD.
The initial consumer is the real Redbrand cave crate/panel oracle in physics-engine.

Cargo-native checks cover analytic signed floor contacts from both sides and windings,
finite triangle boundaries, prediction/input errors, 512 rotated-box overlap comparisons
with the separate polytope/triangle SAT implementation, and BVH versus exhaustive
triangle queries with bounded local work. These do not establish consumer performance
parity; that remains the engine's native/WASM Rapier gate.
