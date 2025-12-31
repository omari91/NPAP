from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import solve, LinAlgError

from npap.exceptions import PartitioningError, ValidationError
from npap.interfaces import PartitioningStrategy
from npap.logging import log_debug, log_info, log_warning, LogCategory
from npap.utils import (
    with_runtime_config,
    create_partition_map, validate_partition,
    run_kmeans, run_kmedoids, validate_required_attributes
)


@dataclass
class ElectricalDistanceConfig:
    """
    Configuration parameters for electrical distance calculations.

    Centralizes all magic numbers and tolerances used in the electrical
    distance partitioning algorithm for better maintainability and tuning.

    Attributes:
        zero_reactance_replacement: Reactance value used when edge reactance is zero.
        regularization_factor: Small value added to B matrix diagonal for numerical stability.
                              Set to 0.0 to disable regularization. Default 1e-10 provides
                              mild regularization that prevents singular matrix issues.
        infinite_distance: Value used to represent "infinite" distance between DC islands.
    """
    zero_reactance_replacement: float = 1e-5
    regularization_factor: float = 1e-10
    infinite_distance: float = 1e4


class ElectricalDistancePartitioning(PartitioningStrategy):
    """
    Partition nodes based on electrical distance using PTDF in power networks.

    This strategy uses the Power Transfer Distribution Factor (PTDF) approach
    to calculate electrical distances between nodes. The electrical distance
    is derived from the Euclidean distance between PTDF column vectors, which
    represents how similarly power injections at different nodes affect line flows.

    Mathematical basis:
        - Build incidence matrix K from directed network topology
        - Remove slack bus column to get K_sba (slack-bus-adjusted)
        - Calculate susceptance matrix B = K_sba^T · diag{b} · K_sba
        - Compute PTDF = diag{b} · K_sba · B^(-1)
        - Electrical distance: d_ij = ||PTDF[:,i] - PTDF[:,j]||_2

    DC Island Isolation:
        Nodes in different DC islands are assigned infinite distance to ensure
        clustering respects DC island boundaries. This is mandatory and requires
        the 'dc_island' attribute on all nodes. Use 'va_loader' data loading
        strategy to automatically detect DC islands, or provide the attribute manually.

    Configuration can be provided at:
        - Instantiation time (via `config` parameter in __init__)
        - Partition time (via `config` or individual parameters in partition())

    Partition-time parameters override instance defaults for that call only.
    """

    SUPPORTED_ALGORITHMS = ['kmeans', 'kmedoids']

    # Config parameter names for runtime override detection
    _CONFIG_PARAMS = {
        'zero_reactance_replacement',
        'regularization_factor',
        'infinite_distance'
    }

    def __init__(self, algorithm: str = 'kmeans', slack_bus: Optional[Any] = None,
                 dc_island_attr: str = 'dc_island',
                 config: Optional[ElectricalDistanceConfig] = None):
        """
        Initialize electrical distance partitioning strategy.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'kmedoids')
            slack_bus: Specific node to use as slack bus, or None for auto-selection
            dc_island_attr: Node attribute name containing DC island ID (default: 'dc_island')
            config: Configuration parameters for distance calculations

        Raises:
            ValueError: If unsupported algorithm is specified
        """
        self.algorithm = algorithm
        self.slack_bus = slack_bus
        self.dc_island_attr = dc_island_attr
        self.config = config or ElectricalDistanceConfig()

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        log_debug(
            f"Initialized ElectricalDistancePartitioning: algorithm={algorithm}, "
            f"dc_island_attr={dc_island_attr}",
            LogCategory.PARTITIONING
        )

    @property
    def required_attributes(self) -> Dict[str, List[str]]:
        """Required attributes for electrical distance partitioning."""
        return {
            'nodes': [],  # dc_island is validated separately with helpful message
            'edges': ['x']  # Reactance attribute required on edges
        }

    def _get_strategy_name(self) -> str:
        """Get descriptive strategy name for error messages."""
        return f"electrical_{self.algorithm}"

    @with_runtime_config(ElectricalDistanceConfig, _CONFIG_PARAMS)
    @validate_required_attributes
    def partition(self, graph: nx.DiGraph, **kwargs) -> Dict[int, List[Any]]:
        """
        Partition nodes based on electrical distance using PTDF.

        Args:
            graph: NetworkX DiGraph with reactance data on edges and dc_island on nodes
            **kwargs: Additional parameters
                - n_clusters: Number of clusters (required)
                - random_state: Random seed for reproducibility
                - max_iter: Maximum iterations for clustering
                - config: ElectricalDistanceConfig instance to override instance config
                - slack_bus: Override the slack bus for this partition call
                - zero_reactance_replacement: Override config parameter
                - regularization_factor: Override config parameter
                - infinite_distance: Override config parameter

        Returns:
            Dictionary mapping cluster_id -> list of node_ids

        Raises:
            PartitioningError: If partitioning fails
            ValidationError: If dc_island attribute is missing
        """
        try:
            # Get effective config (injected by decorator)
            effective_config = kwargs.get('_effective_config', self.config)

            # Resolve slack bus (kwargs override instance default)
            effective_slack = kwargs.get('slack_bus', self.slack_bus)

            # Validate DC island attributes
            self._validate_dc_island_attributes(graph)

            # Validate network connectivity
            self._validate_network_connectivity(graph)

            n_clusters = kwargs.get('n_clusters')

            log_info(
                f"Starting electrical distance partitioning (PTDF): {self.algorithm}, "
                f"n_clusters={n_clusters}",
                LogCategory.PARTITIONING
            )

            if n_clusters is None or n_clusters <= 0:
                raise PartitioningError(
                    "Electrical distance partitioning requires a positive 'n_clusters' parameter.",
                    strategy=self._get_strategy_name()
                )

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            if n_clusters > n_nodes:
                raise PartitioningError(
                    f"Cannot create {n_clusters} clusters from {n_nodes} nodes.",
                    strategy=self._get_strategy_name()
                )

            # Calculate electrical distance matrix using PTDF
            log_debug("Computing PTDF-based electrical distance matrix", LogCategory.PARTITIONING)
            distance_matrix = self._calculate_electrical_distance_matrix(
                graph, nodes, effective_config, effective_slack
            )

            # Perform clustering
            labels = self._run_clustering(distance_matrix, **kwargs)

            # Create and validate partition
            partition_map = create_partition_map(nodes, labels)
            validate_partition(partition_map, n_nodes, self._get_strategy_name())

            # Validate DC island consistency
            self._validate_cluster_dc_island_consistency(graph, partition_map)

            log_info(
                f"Electrical partitioning complete: {len(partition_map)} clusters",
                LogCategory.PARTITIONING
            )

            return partition_map

        except Exception as e:
            if isinstance(e, (PartitioningError, ValidationError)):
                raise
            raise PartitioningError(
                f"Electrical distance partitioning failed: {e}",
                strategy=self._get_strategy_name(),
                graph_info={'nodes': len(list(graph.nodes())), 'edges': len(graph.edges())}
            ) from e

    def _run_clustering(self, distance_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Dispatch to appropriate clustering algorithm."""
        n_clusters = kwargs.get('n_clusters')
        random_state = kwargs.get('random_state', 42)
        max_iter = kwargs.get('max_iter', 300)

        if self.algorithm == 'kmeans':
            log_debug("Running K-means on distance matrix", LogCategory.PARTITIONING)
            return run_kmeans(distance_matrix, n_clusters, random_state, max_iter)
        elif self.algorithm == 'kmedoids':
            log_debug("Running K-medoids on distance matrix", LogCategory.PARTITIONING)
            return run_kmedoids(distance_matrix, n_clusters)
        else:
            raise PartitioningError(
                f"Unknown algorithm: {self.algorithm}",
                strategy=self._get_strategy_name()
            )

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def _validate_dc_island_attributes(self, graph: nx.DiGraph) -> None:
        """
        Validate that all nodes have the DC island attribute.

        Provides a helpful error message directing users to use va_loader
        or manually add the dc_island attribute.

        Args:
            graph: NetworkX DiGraph to validate

        Raises:
            ValidationError: If any node is missing the dc_island attribute
        """
        missing_nodes = []
        for node in graph.nodes():
            if self.dc_island_attr not in graph.nodes[node]:
                missing_nodes.append(node)

        if missing_nodes:
            sample = missing_nodes[:5]
            raise ValidationError(
                f"Electrical distance partitioning requires '{self.dc_island_attr}' attribute "
                f"on all nodes for DC island isolation. "
                f"{len(missing_nodes)} node(s) are missing this attribute (first few: {sample}). "
                f"Please use 'va_loader' data loading strategy to automatically detect DC islands, "
                f"or manually add the '{self.dc_island_attr}' attribute to all nodes.",
                missing_attributes={'nodes': [self.dc_island_attr]},
                strategy=self._get_strategy_name()
            )

    def _validate_network_connectivity(self, graph: nx.DiGraph) -> None:
        """
        Validate network connectivity using weak connectivity for directed graphs.

        For electrical networks, we check weak connectivity (ignoring edge direction)
        because electrical coupling exists regardless of defined edge direction.

        Args:
            graph: NetworkX DiGraph to validate

        Raises:
            PartitioningError: If graph is not weakly connected
        """
        if not nx.is_weakly_connected(graph):
            n_components = nx.number_weakly_connected_components(graph)
            raise PartitioningError(
                f"Graph must be connected for electrical distance partitioning. "
                f"Found {n_components} disconnected components (islands). ",
                strategy=self._get_strategy_name(),
                graph_info={
                    'nodes': len(list(graph.nodes())),
                    'edges': len(graph.edges()),
                    'n_components': n_components
                }
            )

    def _validate_cluster_dc_island_consistency(self, graph: nx.DiGraph,
                                                partition_map: Dict[int, List[Any]]) -> None:
        """
        Validate that clusters don't mix different DC islands.

        With infinite distances between DC islands, clusters should never mix
        nodes from different DC islands.

        Args:
            graph: Original NetworkX graph
            partition_map: Resulting partition mapping
        """
        for cluster_id, nodes in partition_map.items():
            dc_islands_in_cluster = set()

            for node in nodes:
                dc_island = graph.nodes[node].get(self.dc_island_attr)
                if dc_island is not None:
                    dc_islands_in_cluster.add(dc_island)

            if len(dc_islands_in_cluster) > 1:
                log_warning(
                    f"Cluster {cluster_id} contains nodes from multiple DC islands: "
                    f"{dc_islands_in_cluster}. This should not happen with infinite distances.",
                    LogCategory.PARTITIONING,
                    warn_user=False
                )

    # =========================================================================
    # ELECTRICAL DISTANCE CALCULATION METHODS
    # =========================================================================

    def _calculate_electrical_distance_matrix(self, graph: nx.DiGraph, nodes: List[Any],
                                              config: ElectricalDistanceConfig,
                                              slack_bus: Optional[Any]) -> np.ndarray:
        """
        Calculate electrical distance matrix between all node pairs using PTDF.

        Orchestrates the calculation by:
        1. Building the PTDF matrix
        2. Computing Euclidean distances between PTDF columns
        3. Integrating slack bus distances
        4. Applying DC island isolation

        Args:
            graph: NetworkX DiGraph with reactance on edges
            nodes: Ordered list of nodes
            config: ElectricalDistanceConfig instance
            slack_bus: Optional specified slack bus

        Returns:
            Symmetric distance matrix (n_nodes × n_nodes)

        Raises:
            PartitioningError: If distance matrix calculation fails
        """
        selected_slack = self._select_slack_bus(graph, nodes, slack_bus)
        log_debug(f"Selected slack bus: {selected_slack}", LogCategory.PARTITIONING)

        # Build PTDF matrix
        ptdf_matrix, active_nodes = self._build_ptdf_matrix(graph, nodes, selected_slack, config)
        log_debug(f"Built PTDF matrix: shape {ptdf_matrix.shape}", LogCategory.PARTITIONING)

        # Calculate distances from PTDF columns
        distance_matrix_active = self._compute_ptdf_distances(ptdf_matrix)

        # Integrate slack bus into full distance matrix
        distance_matrix_full = self._integrate_slack_bus_distances(
            distance_matrix_active, ptdf_matrix, nodes, selected_slack, active_nodes
        )

        # Apply DC island isolation
        dc_islands = self._extract_dc_islands(graph, nodes)
        distance_matrix_full = self._apply_dc_island_isolation(
            distance_matrix_full, dc_islands, config
        )

        return distance_matrix_full

    def _select_slack_bus(self, graph: nx.DiGraph, nodes: List[Any],
                          slack_bus: Optional[Any]) -> Any:
        """
        Select slack bus node.

        If slack_bus is specified, use it. Otherwise, select the node
        with the highest total degree (in-degree + out-degree).

        Args:
            graph: NetworkX DiGraph
            nodes: List of nodes
            slack_bus: Optional specified slack bus

        Returns:
            Selected slack bus node

        Raises:
            PartitioningError: If specified slack bus not found
        """
        if slack_bus is not None:
            if slack_bus not in nodes:
                raise PartitioningError(
                    f"Specified slack bus {slack_bus} not found in graph.",
                    strategy=self._get_strategy_name()
                )
            return slack_bus

        # Use total degree (in + out) for directed graphs
        degrees = {n: graph.in_degree(n) + graph.out_degree(n) for n in nodes}
        return max(nodes, key=lambda n: degrees[n])

    # =========================================================================
    # PTDF MATRIX CONSTRUCTION
    # =========================================================================

    def _build_ptdf_matrix(self, graph: nx.DiGraph, nodes: List[Any],
                           slack_bus: Any, config: ElectricalDistanceConfig
                           ) -> Tuple[np.ndarray, List[Any]]:
        """
        Build the Power Transfer Distribution Factor (PTDF) matrix.

        PTDF = diag{b} · K_sba · (K_sba^T · diag{b} · K_sba)^(-1)

        Where:
            - K_sba is the slack-bus-adjusted incidence matrix
            - b is the vector of susceptances (1/reactance)
            - The resulting PTDF has shape (n_edges × n_active_nodes)

        Instead of computing B^(-1) explicitly, we solve the linear system
        directly which is significantly faster for large networks.

        Args:
            graph: NetworkX DiGraph with reactance on edges
            nodes: Ordered list of all nodes
            slack_bus: Node designated as slack bus
            config: ElectricalDistanceConfig instance

        Returns:
            Tuple of (PTDF matrix [n_edges × n_active], list of active nodes)

        Raises:
            PartitioningError: If matrix construction fails
        """
        try:
            # Extract edges and susceptances
            edges, susceptances = self._extract_edge_susceptances(graph, config)

            if len(edges) == 0:
                raise PartitioningError(
                    "No valid edges found for PTDF matrix construction.",
                    strategy=self._get_strategy_name()
                )

            # Build slack-bus-adjusted incidence matrix
            K_sba, active_nodes = self._build_incidence_matrix(edges, nodes, slack_bus)
            log_debug(f"Built incidence matrix K_sba: shape {K_sba.shape}", LogCategory.PARTITIONING)

            # Build susceptance matrix B = K_sba^T @ diag(b) @ K_sba
            B_matrix = self._compute_B_matrix(K_sba, susceptances)
            log_debug(f"Built B matrix: shape {B_matrix.shape}", LogCategory.PARTITIONING)

            # Compute PTDF using direct linear solve
            ptdf_matrix = self._compute_ptdf(K_sba, susceptances, B_matrix, config)

            return ptdf_matrix, active_nodes

        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            raise PartitioningError(
                f"Failed to build PTDF matrix: {e}",
                strategy=self._get_strategy_name()
            ) from e

    def _extract_edge_susceptances(self, graph: nx.DiGraph,
                                   config: ElectricalDistanceConfig
                                   ) -> Tuple[List[Tuple[Any, Any]], np.ndarray]:
        """
        Extract directed edges and their susceptances from the graph.

        Each directed edge is treated as a unique electrical element.
        Susceptance b = 1/x where x is the reactance.

        Args:
            graph: NetworkX DiGraph with 'x' (reactance) attribute on edges
            config: ElectricalDistanceConfig instance

        Returns:
            Tuple of (list of (from, to) edges, array of susceptances)

        Raises:
            PartitioningError: If reactance values are invalid
        """
        edges = []
        susceptances = []
        zero_reactance_count = 0

        for u, v, data in graph.edges(data=True):
            reactance = data.get('x')

            if not isinstance(reactance, (int, float)):
                raise PartitioningError(
                    f"Edge ({u}, {v}) reactance must be numeric, got {type(reactance)}",
                    strategy=self._get_strategy_name()
                )

            if reactance == 0:
                zero_reactance_count += 1
                reactance = config.zero_reactance_replacement

            edges.append((u, v))
            susceptances.append(1.0 / reactance)

        # Emit single warning if any zero reactance edges found
        if zero_reactance_count > 0:
            log_warning(
                f"{zero_reactance_count} edge(s) have zero reactance. "
                f"Using replacement value: {config.zero_reactance_replacement}",
                LogCategory.PARTITIONING
            )

        return edges, np.array(susceptances)

    @staticmethod
    def _build_incidence_matrix(edges: List[Tuple[Any, Any]], nodes: List[Any],
                                slack_bus: Any) -> Tuple[np.ndarray, List[Any]]:
        """
        Build slack-bus-adjusted incidence matrix K_sba.

        For each directed edge (u → v):
            - K[edge_idx, u] = -1 (edge leaves u)
            - K[edge_idx, v] = +1 (edge enters v)

        The slack bus column is removed to make B invertible.

        Args:
            edges: List of (from_node, to_node) tuples
            nodes: Ordered list of all nodes
            slack_bus: Node to exclude (slack bus)

        Returns:
            Tuple of (K_sba matrix [n_edges × n_active], list of active nodes)
        """
        # Active nodes (without slack bus)
        active_nodes = [n for n in nodes if n != slack_bus]
        n_active = len(active_nodes)
        n_edges = len(edges)

        # Create node index mapping for active nodes
        node_to_idx = {node: idx for idx, node in enumerate(active_nodes)}

        # Build incidence matrix directly as dense array
        K_sba = np.zeros((n_edges, n_active))

        for edge_idx, (u, v) in enumerate(edges):
            # Edge leaves u: K[edge, u] = -1
            if u in node_to_idx:
                K_sba[edge_idx, node_to_idx[u]] = -1.0

            # Edge enters v: K[edge, v] = +1
            if v in node_to_idx:
                K_sba[edge_idx, node_to_idx[v]] = 1.0

        return K_sba, active_nodes

    @staticmethod
    def _compute_B_matrix(K_sba: np.ndarray, susceptances: np.ndarray) -> np.ndarray:
        """
        Compute susceptance matrix B = K_sba^T @ diag(b) @ K_sba.

        Uses efficient broadcasting: B = (K^T * b) @ K, avoiding explicit
        diagonal matrix construction.

        Args:
            K_sba: Slack-bus-adjusted incidence matrix (n_edges × n_active)
            susceptances: Array of susceptance values (n_edges,)

        Returns:
            Symmetric susceptance matrix (n_active × n_active)
        """
        # Efficient: (K.T * b) @ K where b is broadcast along columns
        # K.T shape: (n_active, n_edges), susceptances shape: (n_edges,)
        K_scaled = K_sba.T * susceptances
        B_matrix = K_scaled @ K_sba

        # Ensure symmetry (handles floating point errors)
        return (B_matrix + B_matrix.T) / 2.0

    @staticmethod
    def _compute_ptdf(K_sba: np.ndarray, susceptances: np.ndarray,
                      B_matrix: np.ndarray,
                      config: ElectricalDistanceConfig) -> np.ndarray:
        """
        Compute PTDF matrix by solving linear system directly.

        To avoid computing B^(-1) explicitly, we solve:
            B @ X = A^T  where A = diag(b) @ K_sba
        Then PTDF = X^T

        This is mathematically equivalent but 3-5x faster for large matrices.

        Args:
            K_sba: Slack-bus-adjusted incidence matrix (n_edges × n_active)
            susceptances: Array of susceptance values (n_edges,)
            B_matrix: Susceptance matrix (n_active × n_active)
            config: ElectricalDistanceConfig instance

        Returns:
            PTDF matrix (n_edges × n_active)
        """
        # Apply Tikhonov regularization for numerical stability
        if config.regularization_factor > 0:
            B_matrix = B_matrix + config.regularization_factor * np.eye(B_matrix.shape[0])

        # A = diag(b) @ K_sba, shape (n_edges, n_active)
        A = susceptances[:, np.newaxis] * K_sba

        try:
            # Solve B @ X = A^T for X, then PTDF = X^T
            # Using assume_a='sym' tells scipy B is symmetric -> uses faster algorithm
            X = solve(B_matrix, A.T, assume_a='sym')
            ptdf_matrix = X.T

        except LinAlgError:
            # Fallback to least-squares solution if matrix is singular
            log_warning(
                "B matrix is singular, using least-squares solution.",
                LogCategory.PARTITIONING
            )
            X, _, _, _ = np.linalg.lstsq(B_matrix, A.T, rcond=None)
            ptdf_matrix = X.T

        return ptdf_matrix

    # =========================================================================
    # DISTANCE CALCULATION METHODS
    # =========================================================================

    def _compute_ptdf_distances(self, ptdf_matrix: np.ndarray) -> np.ndarray:
        """
        Compute electrical distances using Euclidean distance between PTDF columns.

        Electrical distance: d_ij = ||PTDF[:,i] - PTDF[:,j]||_2

        Uses the identity ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ to compute all
        pairwise distances via a single matrix multiplication (Gram matrix),
        which is highly optimized by BLAS libraries.

        Args:
            ptdf_matrix: PTDF matrix (n_edges × n_active)

        Returns:
            Distance matrix for active nodes (n_active × n_active)

        Raises:
            PartitioningError: If distance calculation produces invalid values
        """
        # Use float32 for faster computation
        X = ptdf_matrix.T.astype(np.float32, copy=False)

        # Compute squared norms for each node's PTDF profile
        norms_sq = np.einsum('ij,ij->i', X, X)

        # Compute Gram matrix (all pairwise dot products) - BLAS optimized
        gram = X @ X.T

        # ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        dist_sq = norms_sq[:, np.newaxis] + norms_sq
        dist_sq -= 2.0 * gram  # In-place to save memory

        # Handle numerical errors (small negative values from floating point)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # Take square root to get actual distances
        distance_matrix = np.sqrt(dist_sq, out=dist_sq)  # In-place

        # Ensure diagonal is exactly zero
        np.fill_diagonal(distance_matrix, 0.0)

        if np.any(np.isnan(distance_matrix)):
            raise PartitioningError(
                "Distance matrix contains NaN values after PTDF distance calculation.",
                strategy=self._get_strategy_name()
            )

        return distance_matrix.astype(np.float64)

    def _integrate_slack_bus_distances(self, distance_matrix_active: np.ndarray,
                                       ptdf_matrix: np.ndarray,
                                       nodes: List[Any], slack_bus: Any,
                                       active_nodes: List[Any]) -> np.ndarray:
        """
        Integrate slack bus into the full distance matrix.

        The slack bus has an implicit PTDF column of zeros (reference bus).
        Distance from slack to node i = ||PTDF[:,i] - 0||_2 = ||PTDF[:,i]||_2

        Args:
            distance_matrix_active: Distance matrix for active nodes (n_active × n_active)
            ptdf_matrix: PTDF matrix (n_edges × n_active)
            nodes: Complete list of all nodes
            slack_bus: Slack bus node
            active_nodes: List of active nodes (excluding slack bus)

        Returns:
            Full distance matrix including slack bus (n_nodes × n_nodes)
        """
        n_nodes = len(nodes)

        # Create full matrix
        distance_matrix_full = np.zeros((n_nodes, n_nodes))

        # Get slack index in full node list
        slack_idx = nodes.index(slack_bus)

        # Map active indices to full indices
        active_to_full = [nodes.index(n) for n in active_nodes]

        # Copy active distances to full matrix
        for i, full_i in enumerate(active_to_full):
            for j, full_j in enumerate(active_to_full):
                distance_matrix_full[full_i, full_j] = distance_matrix_active[i, j]

        # Calculate slack bus distances
        # Distance from slack to node i = ||PTDF[:,i]||_2 (L2 norm of column i)
        for i, full_i in enumerate(active_to_full):
            ptdf_column_norm = np.linalg.norm(ptdf_matrix[:, i])
            distance_matrix_full[slack_idx, full_i] = ptdf_column_norm
            distance_matrix_full[full_i, slack_idx] = ptdf_column_norm

        # Validate final matrix
        if np.any(np.isnan(distance_matrix_full)):
            raise PartitioningError(
                "Distance matrix contains NaN values after slack bus integration.",
                strategy=self._get_strategy_name()
            )

        return distance_matrix_full

    # =========================================================================
    # DC ISLAND ISOLATION METHODS
    # =========================================================================

    def _extract_dc_islands(self, graph: nx.DiGraph, nodes: List[Any]) -> np.ndarray:
        """
        Extract DC island IDs for all nodes.

        Args:
            graph: NetworkX DiGraph with dc_island attribute on nodes
            nodes: Ordered list of nodes

        Returns:
            Array of DC island IDs
        """
        return np.array([graph.nodes[node].get(self.dc_island_attr) for node in nodes])

    @staticmethod
    def _apply_dc_island_isolation(distance_matrix: np.ndarray,
                                   dc_islands: np.ndarray,
                                   config: ElectricalDistanceConfig) -> np.ndarray:
        """
        Apply DC island isolation by setting infinite distance between different islands.

        This ensures that clustering algorithms will never group nodes from
        different DC islands into the same cluster.

        Args:
            distance_matrix: Original distance matrix (n_nodes × n_nodes)
            dc_islands: Array of DC island IDs for each node
            config: ElectricalDistanceConfig instance

        Returns:
            Modified distance matrix with infinite distances between DC islands
        """
        # Create mask where True indicates different islands
        island_matrix = dc_islands[:, np.newaxis]
        different_islands = island_matrix != dc_islands

        # Count isolated pairs (upper triangle only to avoid double counting)
        isolation_count = np.sum(np.triu(different_islands, k=1))

        # Apply infinite distance where islands differ
        distance_matrix = np.where(different_islands, config.infinite_distance, distance_matrix)

        if isolation_count > 0:
            n_islands = len(set(dc_islands))
            log_info(
                f"DC island isolation applied: {n_islands} island(s), "
                f"{isolation_count} node pair(s) set to infinite distance",
                LogCategory.PARTITIONING
            )

        return distance_matrix
