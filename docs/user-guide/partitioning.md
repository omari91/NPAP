# Partitioning

Partitioning divides a network into clusters of nodes based on distance metrics. NPAP provides four partitioning strategy families, each suited for different use cases.

## Overview

| Strategy Family | Distance Metric      | Use Case |
|-----------------|----------------------|----------|
| Geographic | Euclidean / Haversine | Spatial proximity clustering |
| Electrical | PTDF-based           | Electrical behavior clustering |
| VA Geographic | Geographic + Voltage | Multi-voltage networks |
| VA Electrical | PTDF + Voltage aware | Multi-voltage electrical clustering |

## Basic Usage

All partitioning is done through the {py:class}`~npap.PartitionAggregatorManager`:

```python
import npap

manager = npap.PartitionAggregatorManager()
manager.load_data("networkx_direct", graph=G)

# Partition the network
partition_result = manager.partition(
    strategy="geographical_kmeans",
    n_clusters=10
)
```

The result is a {py:class}`~npap.PartitionResult` object:

```python
print(partition_result.mapping)
# {0: ['bus_1', 'bus_2'], 1: ['bus_3', 'bus_4'], ...}

print(partition_result.n_clusters)
# 10

print(partition_result.strategy_name)
# 'geographical_kmeans'
```

## Geographical Partitioning

Clusters nodes based on geographic coordinates using various clustering algorithms.

### Required Attributes

- **Nodes**: `lat` (latitude), `lon` (longitude)

### Available Strategies

| Strategy | Algorithm | Metric | DC-Island Aware |
|----------|-----------|--------|-----------------|
| `geographical_kmeans` | K-Means | Euclidean | No |
| `geographical_kmedoids_euclidean` | K-Medoids | Euclidean | Yes |
| `geographical_kmedoids_haversine` | K-Medoids | Haversine | Yes |
| `geographical_dbscan_euclidean` | DBSCAN | Euclidean | Yes |
| `geographical_dbscan_haversine` | DBSCAN | Haversine | Yes |
| `geographical_hierarchical` | Agglomerative | Euclidean | Yes* |
| `geographical_hdbscan_euclidean` | HDBSCAN | Euclidean | Yes |
| `geographical_hdbscan_haversine` | HDBSCAN | Haversine | Yes |

*Ward linkage does not support DC-island awareness

### K-Means Clustering

Fast clustering on raw coordinates. Best for quick partitioning when DC-island boundaries don't matter.

```python
partition = manager.partition(
    "geographical_kmeans",
    n_clusters=10,
    random_state=42,
    max_iter=300,
    n_init=10
)
```

**Parameters:**
- `n_clusters`: Number of clusters (required)
- `random_state`: Random seed for reproducibility
- `max_iter`: Maximum iterations
- `n_init`: Number of initializations

```{warning}
K-Means does **not** support DC-island awareness. If your network has DC links, consider using K-Medoids instead.
```

### K-Medoids Clustering

More robust clustering using precomputed distance matrices. Supports DC-island constraints.

```python
# Euclidean distance (for projected coordinates)
partition = manager.partition(
    "geographical_kmedoids_euclidean",
    n_clusters=10
)

# Haversine distance (for lat/lon on Earth's surface)
partition = manager.partition(
    "geographical_kmedoids_haversine",
    n_clusters=10
)
```
**Parameters:**
- `n_clusters`: Number of clusters (required)

### DBSCAN Clustering

Density-based clustering that automatically determines the number of clusters.

```python
partition = manager.partition(
    "geographical_dbscan_euclidean",
    eps=0.5,          # Maximum distance between points in a cluster
    min_samples=5     # Minimum points to form a cluster
)

print(f"Found {partition.n_clusters} clusters")
```

**Parameters:**
- `eps`: Maximum distance between two samples in the same neighborhood
- `min_samples`: Minimum number of samples in a neighborhood for a point to be a core point

```{note}
DBSCAN may classify some nodes as noise (cluster -1). These are typically isolated nodes.
```

### Hierarchical Clustering

Agglomerative clustering with different linkage methods.

```python
partition = manager.partition(
    "geographical_hierarchical",
    n_clusters=10,
    linkage="complete"  # 'complete', 'average', 'single'
)
```

**Parameters:**
- `n_clusters`: Number of clusters
- `linkage`: Linkage criterion
  - `complete`: Maximum distance between clusters
  - `average`: Average distance between clusters
  - `single`: Minimum distance between clusters

```{warning}
Ward linkage (`ward`) is **not supported** when DC-island awareness is needed, as it requires Euclidean distances and cannot use precomputed distance matrices.
```

### HDBSCAN Clustering

Hierarchical density-based clustering for robust automatic cluster detection.

```python
partition = manager.partition(
    "geographical_hdbscan_haversine",
    min_cluster_size=5
)
```

**Parameters:**
- `min_cluster_size`: Minimum size of clusters

## Electrical Partitioning

Clusters nodes based on **electrical distance** computed from Power Transfer Distribution Factors (PTDF).

### Required Attributes

- **Nodes**: `dc_island` (DC island identifier)
- **Edges**: `x` (reactance)

### Mathematical Background

Electrical distance is based on how power flows through the network:

1. **Incidence Matrix (K)**: Describes network topology
2. **Susceptance Matrix (B)**: $B = K_{sba}^T \cdot \text{diag}(b) \cdot K_{sba}$
3. **PTDF Matrix**: $PTDF = \text{diag}(b) \cdot K_{sba} \cdot B^{-1}$
4. **Electrical Distance**: $d_{ij} = ||PTDF_{:,i} - PTDF_{:,j}||_2$

Nodes with similar PTDF columns have similar electrical behavior.

### Available Strategies

| Strategy | Algorithm | Description |
|----------|-----------|-------------|
| `electrical_kmeans` | K-Means | Fast electrical clustering |
| `electrical_kmedoids` | K-Medoids | Robust electrical clustering |

### Basic Usage

```python
partition = manager.partition(
    "electrical_kmeans",
    n_clusters=10
)
```

### DC-Island Handling

Electrical partitioning is always DC-island aware:

- PTDF is computed independently for each DC island
- Nodes in different islands have infinite distance
- Each island is clustered separately

```{mermaid}
flowchart TB
    subgraph Island0[DC Island 0]
        A[Cluster 0] --- B[Cluster 1]
    end

    subgraph Island1[DC Island 1]
        C[Cluster 2] --- D[Cluster 3]
    end

    Island0 -.->|DC Link| Island1

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
```

### Configuration

```python
from npap.partitioning import ElectricalDistanceConfig

# Custom configuration
config = ElectricalDistanceConfig(
    zero_reactance_replacement=1e-5,  # Replace zero reactance
    regularization_factor=1e-10,       # Matrix regularization
    infinite_distance=1e4              # Inter-island distance
)

partition = manager.partition(
    "electrical_kmeans",
    n_clusters=10,
    config=config
)
```

## Voltage-Aware Geographical Partitioning

Combines geographic distance with voltage level and DC island constraints.

### Required Attributes

- **Nodes**: `lat`, `lon`, `voltage`, `dc_island`

### Available Strategies

| Strategy | Mode | Description |
|----------|------|-------------|
| `va_geographical_kmedoids_euclidean` | Standard | K-Medoids with Euclidean distance |
| `va_geographical_kmedoids_haversine` | Standard | K-Medoids with Haversine distance |
| `va_geographical_hierarchical` | Standard | Agglomerative clustering |
| `va_geographical_proportional_kmedoids_euclidean` | Proportional | Balanced clustering by group |
| `va_geographical_proportional_kmedoids_haversine` | Proportional | Balanced clustering by group |
| `va_geographical_proportional_hierarchical` | Proportional | Balanced hierarchical |

### Standard Mode

Nodes in different voltage levels or DC islands receive infinite distance:

```python
partition = manager.partition(
    "va_geographical_kmedoids_haversine",
    n_clusters=20
)
```

### Proportional Mode

Distributes clusters proportionally among voltage/island groups:

```python
# If network has:
# - 100 nodes at 380 kV
# - 50 nodes at 220 kV
# And n_clusters=15, approximately:
# - 10 clusters for 380 kV nodes
# - 5 clusters for 220 kV nodes

partition = manager.partition(
    "va_geographical_proportional_kmedoids_haversine",
    n_clusters=15
)
```

### Configuration

```python
from npap.partitioning import VAGeographicalConfig

config = VAGeographicalConfig(
    voltage_tolerance=1.0,        # kV tolerance for grouping
    infinite_distance=1e4,
    proportional_clustering=False,
    hierarchical_linkage="complete"
)

partition = manager.partition(
    "va_geographical_kmedoids_haversine",
    n_clusters=20,
    config=config
)
```

## Voltage-Aware Electrical Partitioning

Combines electrical distance (PTDF) with voltage level constraints.

### Required Attributes

- **Nodes**: `voltage`, `dc_island`
- **Edges**: `x` (reactance)

### Available Strategies

| Strategy | Algorithm |
|----------|-----------|
| `va_electrical_kmedoids` | K-Medoids |
| `va_electrical_hierarchical` | Agglomerative |

### Basic Usage

```python
partition = manager.partition(
    "va_electrical_kmedoids",
    n_clusters=20
)
```

### How It Works

1. Computes PTDF for the full AC network (lines + transformers)
2. Calculates electrical distances from PTDF
3. Sets infinite distance between nodes at different voltage levels
4. Clusters respecting both electrical behavior and voltage boundaries

```{mermaid}
flowchart TB
    subgraph V380[380 kV Level]
        A[Cluster 0] --- B[Cluster 1]
    end

    subgraph V220[220 kV Level]
        C[Cluster 2] --- D[Cluster 3]
    end

    V380 -->|Transformer| V220

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#2993B5,stroke:#1d6f8a,color:#fff
    style C fill:#0fad6b,stroke:#076b3f,color:#fff
    style D fill:#0fad6b,stroke:#076b3f,color:#fff
```

## DC-Island Awareness

### What Are DC Islands?

In networks with HVDC links, AC-connected regions form **DC islands**. These islands:

- Are electrically separate in AC sense
- Connected only through DC links
- Have independent power flow characteristics

### How NPAP Handles DC Islands

When nodes have a `dc_island` attribute (automatically set by `va_loader`):

1. **Distance matrices** include infinite distances between islands
2. **Clustering algorithms** using precomputed distances respect these boundaries
3. **Partitions** never span across DC island boundaries

### Algorithm Support

| Algorithm | DC-Island Support |
|-----------|-------------------|
| K-Means | No (works on raw coordinates) |
| K-Medoids | Yes |
| DBSCAN | Yes |
| HDBSCAN | Yes |
| Hierarchical (non-ward) | Yes |
| Hierarchical (ward) | No |

```{note}
K-Means and Ward linkage will issue a warning if DC islands are detected but will proceed without respecting island boundaries.
```

## Choosing a Strategy

### Decision Guide

```{mermaid}
flowchart TD
    A[Start] --> B{Multi-voltage?}
    B -->|Yes| C{Need electrical distance?}
    B -->|No| D{Need electrical distance?}
    C -->|Yes| E[va_electrical_*]
    C -->|No| F[va_geographical_*]
    D -->|Yes| G[electrical_*]
    D -->|No| H{DC islands?}
    H -->|Yes| I[geographical_kmedoids_*]
    H -->|No| J{Fast or robust?}
    J -->|Fast| K[geographical_kmeans]
    J -->|Robust| L[geographical_kmedoids_*]

    style A fill:#2993B5,stroke:#1d6f8a,color:#fff
    style B fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style C fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style D fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style E fill:#0fad6b,stroke:#076b3f,color:#fff
    style F fill:#0fad6b,stroke:#076b3f,color:#fff
    style G fill:#0fad6b,stroke:#076b3f,color:#fff
    style H fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style I fill:#0fad6b,stroke:#076b3f,color:#fff
    style J fill:#FFBF00,stroke:#cc9900,color:#1e293b
    style K fill:#0fad6b,stroke:#076b3f,color:#fff
    style L fill:#0fad6b,stroke:#076b3f,color:#fff
```

### Recommendations by Use Case

| Use Case | Recommended Strategy |
|----------|---------------------|
| Quick geographic clustering | `geographical_kmeans` |
| Geographic with DC islands | `geographical_kmedoids_haversine` |
| Electrical behavior grouping | `electrical_kmedoids` |
| Multi-voltage network | `va_geographical_kmedoids_haversine` |
| Unknown cluster count | `geographical_dbscan_*` or `geographical_hdbscan_*` |
| Large networks (>10k nodes) | `geographical_kmeans` |

### Performance Comparison

| Strategy | Speed | Memory | Robustness |
|----------|-------|--------|------------|
| K-Means | Fast | Low | Medium |
| K-Medoids | Medium | High (distance matrix) | High |
| DBSCAN | Medium | Medium | High |
| HDBSCAN | Slow | High | Very High |
| Hierarchical | Medium | High | High |

## Configuration Options

All geographical partitioning strategies share a common configuration:

```python
from npap.partitioning import GeographicalConfig

config = GeographicalConfig(
    random_state=42,              # Reproducibility
    max_iter=300,                 # K-Means iterations
    n_init=10,                    # K-Means initializations
    hierarchical_linkage="ward",  # Hierarchical linkage
    infinite_distance=1e4         # DC island separation
)
```

Electrical strategies have their own configuration:

```python
from npap.partitioning import ElectricalDistanceConfig

config = ElectricalDistanceConfig(
    zero_reactance_replacement=1e-5,
    regularization_factor=1e-10,
    infinite_distance=1e4
)
```

## Working with Partition Results

### Inspecting Results

```python
partition = manager.partition("geographical_kmeans", n_clusters=5)

# Cluster mapping
print(partition.mapping)
# {0: ['bus_1', 'bus_2'], 1: ['bus_3'], ...}

# Number of clusters
print(partition.n_clusters)

# Strategy used
print(partition.strategy_name)

# Graph hash for validation
print(partition.original_graph_hash)
```

### Reusing Partitions

Partition results can be reused for multiple aggregations:

```python
# Partition once
partition = manager.partition("geographical_kmeans", n_clusters=10)

# Aggregate with different profiles
simple_agg = manager.aggregate(partition, mode=AggregationMode.SIMPLE)
geo_agg = manager.aggregate(partition, mode=AggregationMode.GEOGRAPHICAL)
```

### Validation

NPAP validates that partitions match their source graph:

```python
# This will raise GraphCompatibilityError if graph changed
aggregated = manager.aggregate(partition)
```
