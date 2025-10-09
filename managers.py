from typing import Dict

import networkx as nx

from .interfaces import (
    DataLoadingStrategy
)


class InputDataManager:
    """Manages data loading from different sources"""

    def __init__(self):
        self._strategies: Dict[str, DataLoadingStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: DataLoadingStrategy):
        """Register a new data loading strategy"""
        self._strategies[name] = strategy

    def load(self, strategy_name: str, **kwargs) -> nx.Graph:
        """Load data using specified strategy"""
        if strategy_name not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown data loading strategy: {strategy_name}. Available: {available}")

        strategy = self._strategies[strategy_name]
        strategy.validate_inputs(**kwargs)
        return strategy.load(**kwargs)

    def _register_default_strategies(self):
        """Register built-in loading strategies"""
        from .input.csv_loader import CSVFilesStrategy
        from .input.networkx_loader import NetworkXDirectStrategy

        self._strategies['csv_files'] = CSVFilesStrategy()
        self._strategies['networkx_direct'] = NetworkXDirectStrategy()
