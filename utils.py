import functools

import networkx as nx


def validate_required_attributes(func):
    """
    Optimized decorator for attribute validation
    - Early exit on first missing attribute
    - Minimal memory overhead
    - Fast attribute checking
    """

    @functools.wraps(func)
    def wrapper(self, graph: nx.Graph, n_clusters: int, **kwargs):
        required = self.required_attributes

        # Fast early-exit validation for nodes
        if required.get('nodes'):
            required_node_attrs = set(required['nodes'])  # Set lookup is O(1)
            for node_id in graph.nodes():
                node_attrs = graph.nodes[node_id]
                # Use set operations - much faster than individual checks
                missing_attrs = required_node_attrs - node_attrs.keys()
                if missing_attrs:
                    from .exceptions import ValidationError
                    raise ValidationError(
                        f"Node {node_id} missing required attributes",
                        missing_attributes={'nodes': list(missing_attrs)},
                        strategy=self.__class__.__name__
                    )

        # Fast early-exit validation for edges
        if required.get('edges'):
            required_edge_attrs = set(required['edges'])
            for edge in graph.edges():
                edge_attrs = graph.edges[edge]
                missing_attrs = required_edge_attrs - edge_attrs.keys()
                if missing_attrs:
                    from .exceptions import ValidationError
                    raise ValidationError(
                        f"Edge {edge} missing required attributes",
                        missing_attributes={'edges': list(missing_attrs)},
                        strategy=self.__class__.__name__
                    )

        # Only call original function if validation passes
        return func(self, graph, n_clusters, **kwargs)

    return wrapper
