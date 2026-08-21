# Arkadev & Braverman — "Computers and Pattern Recognition" (1966)

A reconstruction of experiments described in A. G. Arkadev and E. M.
Braverman's monograph (trans. W. Turski & J. D. Cowan, Thompson Book
Co., 1966). This is a detour from the main 7-stage roadmap (see
`../../01_perceptron/` etc.), not part of it -- it exists because the
book raised questions worth actually running rather than just reading
about, and because the compute to do so simply didn't exist in 1966.

The book's central idea is the **compactness hypothesis**: an "image"
(their term for a class) corresponds to a compact set of points in
receptor space -- few boundary points, smooth borders, no
"peninsulas" reaching into other classes. If true, recognition reduces
to finding separating hypersurfaces from a handful of examples.

Each subfolder reconstructs one chapter's algorithm against real data
(scikit-learn's bundled `load_digits` corpus -- 1797 real handwritten
8x8 digit images, close in scale to the book's own 60-cell examples)
rather than the book's small hand-tabulated experiments.

- `ch2_compactness/` -- receptor-space encoding, internal/boundary point
  definitions, and a direct measurement of whether real digit classes
  are actually compact by the book's own definition.
- `ch3_dissecting_planes/` -- the random-hyperplane learning algorithm
  from Chapter 3 (Tables IV-XIX), reproducing reliability-vs-N curves
  like their Tables XV-XVIII, on real handwriting instead of ~12-20
  hand-prepared representatives per digit.

Later chapters (potential functions, the Perceptron proper, further
improvements) may get the same treatment in additional subfolders if
we come back to them.
