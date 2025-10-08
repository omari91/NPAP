import networkx as nx

from ..exceptions import DataLoadingError
from ..interfaces import DataLoadingStrategy


class NetworkXDirectStrategy(DataLoadingStrategy):
    """Use NetworkX graph directly"""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that a NetworkX graph is provided"""
        if 'graph' not in kwargs:
            raise DataLoadingError(
                "Missing required parameter: graph",
                strategy="networkx_direct",
                details={'required_params': ['graph']}
            )

        graph = kwargs['graph']
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise DataLoadingError(
                f"Parameter 'graph' must be a NetworkX Graph, got {type(graph)}",
                strategy="networkx_direct",
                details={'provided_type': str(type(graph))}
            )

        # Additional validation
        if len(list(graph.nodes())) == 0:
            raise DataLoadingError(
                "Provided graph has no nodes",
                strategy="networkx_direct"
            )

        return True

    def load(self, graph: nx.Graph, **kwargs) -> nx.Graph:
        """Use graph directly (create a copy to avoid mutations)"""
        try:
            # Convert to simple Graph if needed and create copy
            if isinstance(graph, nx.Graph):
                result = graph.copy()
            else:
                # Convert other graph types to simple Graph
                result = nx.Graph()
                result.add_nodes_from(graph.nodes(data=True))
                result.add_edges_from(graph.edges(data=True))

            # Validate the copied graph
            if len(list(result.nodes())) == 0:
                raise DataLoadingError(
                    "Graph has no nodes after processing",
                    strategy="networkx_direct"
                )

            return result

        except Exception as e:
            raise DataLoadingError(
                f"Error processing NetworkX graph: {e}",
                strategy="networkx_direct"
            ) from e
