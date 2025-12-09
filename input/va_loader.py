"""
Voltage-Aware Data Loading Strategy

Loads network data from three separate CSV files:
- nodes.csv: Node/bus data with geographical and electrical properties
- lines.csv: Transmission/distribution line data
- transformers.csv: Transformer data

Combines lines and transformers into a unified edge representation with
voltage awareness for power system analysis.
"""

from pathlib import Path

import networkx as nx
import pandas as pd

from exceptions import DataLoadingError
from interfaces import DataLoadingStrategy


class VoltageAwareStrategy(DataLoadingStrategy):
    """
    Load graph from separate CSV files for nodes, lines, and transformers.

    This strategy creates a directed graph (DiGraph or MultiDiGraph) where:
    - Nodes represent electrical buses/substations
    - Edges represent either transmission lines or transformers
    - Edge direction typically flows from higher to lower voltage (for transformers)
      or follows defined bus0 -> bus1 convention

    Edge Schema:
        - bus0, bus1: Source and target node IDs
        - line_voltage: Voltage level for regular lines (0 for transformers)
        - primary_voltage: Primary voltage for transformers (0 for lines)
        - secondary_voltage: Secondary voltage for transformers (0 for lines)
        - reactance (x): Line/transformer reactance
        - lat, lon: Geographic coordinates (optional)
        - is_trafo: Boolean indicating if edge is a transformer
    """

    # Required columns for each file type
    REQUIRED_NODE_COLUMNS = ['bus_id']  # Minimum required, others are optional
    REQUIRED_LINE_COLUMNS = ['bus0', 'bus1', 'x']  # Reactance required for electrical analysis
    REQUIRED_TRANSFORMER_COLUMNS = ['bus0', 'bus1', 'x', 'primary_voltage', 'secondary_voltage']

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that required CSV files are provided and exist.

        Args:
            **kwargs: Must include 'node_file', 'line_file', 'transformer_file'

        Returns:
            True if validation passes

        Raises:
            DataLoadingError: If validation fails
        """
        required_files = ['node_file', 'line_file', 'transformer_file']
        missing = [f for f in required_files if f not in kwargs or kwargs[f] is None]

        if missing:
            raise DataLoadingError(
                f"Missing required parameters: {missing}",
                strategy="va_loader",
                details={
                    'required_params': required_files,
                    'provided_params': list(kwargs.keys())
                }
            )

        # Check if files exist
        for param in required_files:
            file_path = Path(kwargs[param])
            if not file_path.exists():
                raise DataLoadingError(
                    f"File not found: {file_path}",
                    strategy="va_loader",
                    details={'missing_file': str(file_path)}
                )

        return True

    def load(self, node_file: str, line_file: str, transformer_file: str,
             **kwargs) -> nx.DiGraph | nx.MultiDiGraph:
        """
        Load graph from three CSV files: nodes, lines, and transformers.

        Args:
            node_file: Path to nodes CSV file
            line_file: Path to lines CSV file
            transformer_file: Path to transformers CSV file
            **kwargs: Additional parameters:
                - delimiter: CSV delimiter (default: ',')
                - decimal: Decimal separator (default: '.')
                - node_id_col: Column name for node IDs (auto-detected if not provided)

        Returns:
            DiGraph or MultiDiGraph with combined line and transformer edges

        Raises:
            DataLoadingError: If loading fails
        """
        try:
            delimiter = kwargs.get('delimiter', ',')
            decimal = kwargs.get('decimal', '.')

            # Load and validate nodes
            nodes_df = self._load_nodes(node_file, delimiter, decimal)

            # Load and validate lines
            lines_df = self._load_lines(line_file, delimiter, decimal)

            # Load and validate transformers
            transformers_df = self._load_transformers(transformer_file, delimiter, decimal)

            # Get node ID column
            node_id_col = kwargs.get('node_id_col', self._detect_id_column(nodes_df))
            if node_id_col not in nodes_df.columns:
                raise DataLoadingError(
                    f"Node ID column '{node_id_col}' not found in node file",
                    strategy="va_loader",
                    details={'available_columns': list(nodes_df.columns)}
                )

            # Validate edge references
            valid_node_ids = set(nodes_df[node_id_col].values)
            self._validate_edge_references(lines_df, valid_node_ids, "lines")
            self._validate_edge_references(transformers_df, valid_node_ids, "transformers")

            # Prepare node tuples
            node_tuples = self._prepare_node_tuples(nodes_df, node_id_col)

            # Prepare edge tuples (combine lines and transformers)
            line_tuples = self._prepare_line_tuples(lines_df)
            transformer_tuples = self._prepare_transformer_tuples(transformers_df)
            all_edge_tuples = line_tuples + transformer_tuples

            # Check for parallel edges
            has_parallel_edges = self._check_parallel_edges(all_edge_tuples)

            # Create appropriate graph type
            if has_parallel_edges:
                graph = nx.MultiDiGraph()
                print("MULTI-DIGRAPH DETECTED: Parallel edges found in the data.")
                print("The loaded graph contains multiple edges between the same node pairs.")
                print("MultiDiGraphs cannot be partitioned directly.")
                print("Call manager.aggregate_parallel_edges() to collapse parallel edges.")
            else:
                graph = nx.DiGraph()

            graph.add_nodes_from(node_tuples)
            graph.add_edges_from(all_edge_tuples)

            return graph

        except DataLoadingError:
            raise
        except pd.errors.EmptyDataError as e:
            raise DataLoadingError(f"Empty CSV file: {e}", strategy="va_loader") from e
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parsing error: {e}", strategy="va_loader") from e
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading voltage-aware CSV files: {e}",
                strategy="va_loader"
            ) from e

    @staticmethod
    def _load_nodes(file_path: str, delimiter: str, decimal: str, **kwargs) -> pd.DataFrame:
        """Load and validate nodes DataFrame."""
        nodes_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)

        if nodes_df.empty:
            raise DataLoadingError("Node file is empty", strategy="va_loader")

        return nodes_df

    def _load_lines(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate lines DataFrame."""
        lines_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if lines_df.empty:
            print("Warning: Lines file is empty. Proceeding with transformers only.")
            return pd.DataFrame(columns=self.REQUIRED_LINE_COLUMNS)

        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_LINE_COLUMNS if col not in lines_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Lines file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(lines_df.columns)}
            )

        return lines_df

    def _load_transformers(self, file_path: str, delimiter: str, decimal: str) -> pd.DataFrame:
        """Load and validate transformers DataFrame."""
        transformers_df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, quotechar="'")

        if transformers_df.empty:
            print("Warning: Transformers file is empty. Proceeding with lines only.")
            return pd.DataFrame(columns=self.REQUIRED_TRANSFORMER_COLUMNS)

        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_TRANSFORMER_COLUMNS
                        if col not in transformers_df.columns]
        if missing_cols:
            raise DataLoadingError(
                f"Transformers file missing required columns: {missing_cols}",
                strategy="va_loader",
                details={'available_columns': list(transformers_df.columns)}
            )

        # Validate voltage values for transformers
        for idx, row in transformers_df.iterrows():
            primary_v = row.get('primary_voltage', 0)
            secondary_v = row.get('secondary_voltage', 0)

            if pd.isna(primary_v) or pd.isna(secondary_v):
                raise DataLoadingError(
                    f"Transformer at row {idx} has missing voltage values",
                    strategy="va_loader"
                )

            if primary_v <= 0 or secondary_v <= 0:
                raise DataLoadingError(
                    f"Transformer at row {idx} must have positive primary and secondary voltages",
                    strategy="va_loader",
                    details={
                        'row': idx,
                        'primary_voltage': primary_v,
                        'secondary_voltage': secondary_v
                    }
                )

        return transformers_df

    @staticmethod
    def _validate_edge_references(edges_df: pd.DataFrame, valid_node_ids: set,
                                  edge_type: str) -> None:
        """Validate that all edge node references exist in the node set."""
        if edges_df.empty:
            return

        for idx, row in edges_df.iterrows():
            bus0 = row['bus0']
            bus1 = row['bus1']

            if bus0 not in valid_node_ids:
                raise DataLoadingError(
                    f"{edge_type.capitalize()} at row {idx} references non-existent node: {bus0}",
                    strategy="va_loader"
                )
            if bus1 not in valid_node_ids:
                raise DataLoadingError(
                    f"{edge_type.capitalize()} at row {idx} references non-existent node: {bus1}",
                    strategy="va_loader"
                )

    @staticmethod
    def _prepare_node_tuples(nodes_df: pd.DataFrame, node_id_col: str) -> list:
        """Prepare node tuples for graph creation."""
        return [
            (
                row[node_id_col],
                {col: row[col] for col in nodes_df.columns
                 if col != node_id_col and pd.notna(row[col])}
            )
            for _, row in nodes_df.iterrows()
        ]

    @staticmethod
    def _prepare_line_tuples(lines_df: pd.DataFrame) -> list:
        """
        Prepare line edge tuples with standardized schema.

        Lines have:
        - line_voltage > 0
        - primary_voltage = 0
        - secondary_voltage = 0
        - is_trafo = False
        """
        edge_tuples = []

        for _, row in lines_df.iterrows():
            attrs = {
                'is_trafo': False,
                'line_voltage': row.get('line_voltage', row.get('voltage', 0)),
                'primary_voltage': 0,
                'secondary_voltage': 0,
            }

            # Add all other columns as attributes
            for col in lines_df.columns:
                if col not in ['bus0', 'bus1'] and pd.notna(row[col]):
                    # Don't overwrite standardized fields
                    if col not in attrs:
                        attrs[col] = row[col]

            edge_tuples.append((row['bus0'], row['bus1'], attrs))

        return edge_tuples

    @staticmethod
    def _prepare_transformer_tuples(transformers_df: pd.DataFrame) -> list:
        """
        Prepare transformer edge tuples with standardized schema.

        Transformers have:
        - line_voltage = 0
        - primary_voltage > 0
        - secondary_voltage > 0
        - is_trafo = True
        """
        edge_tuples = []

        for _, row in transformers_df.iterrows():
            attrs = {
                'is_trafo': True,
                'line_voltage': 0,
                'primary_voltage': row['primary_voltage'],
                'secondary_voltage': row['secondary_voltage'],
            }

            # Add all other columns as attributes
            for col in transformers_df.columns:
                if col not in ['bus0', 'bus1', 'primary_voltage', 'secondary_voltage'] \
                        and pd.notna(row[col]):
                    if col not in attrs:
                        attrs[col] = row[col]

            edge_tuples.append((row['bus0'], row['bus1'], attrs))

        return edge_tuples

    @staticmethod
    def _check_parallel_edges(edge_tuples: list) -> bool:
        """Check if there are parallel edges (same source-target pair)."""
        seen_pairs = set()

        for edge in edge_tuples:
            # For directed graphs, (A, B) and (B, A) are different edges
            pair = (edge[0], edge[1])
            if pair in seen_pairs:
                return True
            seen_pairs.add(pair)

        return False

    @staticmethod
    def _detect_id_column(df: pd.DataFrame) -> str:
        """Detect the ID column for nodes."""
        candidates = [
            'node_id', 'nodeId', 'node_ID', 'NodeId',
            'bus_id', 'busId', 'bus_ID',
            'id', 'Id', 'ID', 'index', 'name'
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Fallback to first column
        return df.columns[0]
