from .build_graph import build_similarity_matrix, build_knn_graph
from .builder import GraphBuilder
from .laplacian import compute_laplacian, compute_normalized_laplacian
from .laplacian_builder import LaplacianBuilder

__all__ = [
    "build_similarity_matrix",
    "build_knn_graph",
    "GraphBuilder",
    "compute_laplacian",
    "compute_normalized_laplacian",
    "LaplacianBuilder",
]

# GraphBuilder.build(X) -> (W, deg, adj_indices)
# LaplacianBuilder.build(W) -> L (dense)
# build_knn_graph now accepts as_sparse=True to return a torch.sparse_coo_tensor.
# compute_laplacian / compute_normalized_laplacian accept sparse_coo input.
