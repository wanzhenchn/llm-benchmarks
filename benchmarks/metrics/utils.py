from typing import Optional
import numpy as np


def process_llm_metrics(llm_metrics: Optional[dict]) -> dict:
    """
    Process LLM metrics data, adaptively handling all metrics.

    This function processes all metrics in the llm_metrics dictionary by:
    - Filtering out NaN and None values
    - Removing the first 3 and last 2 data points if there are more than 5 data points
    - Calculating the mean value for each metric

    Special handling for vllm prefix cache metrics:
    If the dictionary contains
    'prefix_cache_hits', 'prefix_cache_queries', 'external_prefix_cache_hits',
    and 'external_prefix_cache_queries', these four metrics are processed
    synchronously (filtering corresponding positions) and two hit rates are calculated:
    - prefix_cache_hit_rate = prefix_cache_hits / prefix_cache_queries
    - external_prefix_cache_hit_rate = external_prefix_cache_hits / external_prefix_cache_queries

    Args:
        llm_metrics: Dictionary of LLM metrics, where keys are metric names
                     and values are lists of data points

    Returns:
        Dictionary of processed metrics, where keys are metric names
        and values are the calculated mean values
    """
    processed_metrics = {}

    if not llm_metrics:
        return processed_metrics

    # Check if all four cache vllm metrics are present
    cache_metric_keys = [
        'prefix_cache_hits',
        'prefix_cache_queries',
        'external_prefix_cache_hits',
        'external_prefix_cache_queries'
    ]

    has_all_cache_metrics = all(key in llm_metrics for key in cache_metric_keys)

    if has_all_cache_metrics:
        # Process cache metrics synchronously
        prefix_hits = llm_metrics['prefix_cache_hits']
        prefix_queries = llm_metrics['prefix_cache_queries']
        external_hits = llm_metrics['external_prefix_cache_hits']
        external_queries = llm_metrics['external_prefix_cache_queries']

        # Synchronously filter out NaN and None values at corresponding positions
        valid_indices = []
        max_len = max(len(prefix_hits), len(prefix_queries),
                     len(external_hits), len(external_queries))

        for i in range(max_len):
            # Check if all values at this position are valid
            values = [
                prefix_hits[i] if i < len(prefix_hits) else None,
                prefix_queries[i] if i < len(prefix_queries) else None,
                external_hits[i] if i < len(external_hits) else None,
                external_queries[i] if i < len(external_queries) else None
            ]

            # Check if all values are valid (not NaN and not None)
            def is_valid_value(v):
                if v is None:
                    return False
                if isinstance(v, float):
                    return not (v != v or np.isnan(v))
                return True

            if all(is_valid_value(v) for v in values):
                valid_indices.append(i)

        if valid_indices:
            # Extract valid data at corresponding positions
            valid_prefix_hits = [prefix_hits[i] for i in valid_indices]
            valid_prefix_queries = [prefix_queries[i] for i in valid_indices]
            valid_external_hits = [external_hits[i] for i in valid_indices]
            valid_external_queries = [external_queries[i] for i in valid_indices]

            # Remove first 3 and last 2 data points if there are more than 5 data points
            if len(valid_indices) > 5:
                filtered_prefix_hits = valid_prefix_hits[3:-2]
                filtered_prefix_queries = valid_prefix_queries[3:-2]
                filtered_external_hits = valid_external_hits[3:-2]
                filtered_external_queries = valid_external_queries[3:-2]
            else:
                filtered_prefix_hits = valid_prefix_hits
                filtered_prefix_queries = valid_prefix_queries
                filtered_external_hits = valid_external_hits
                filtered_external_queries = valid_external_queries

            # Calculate hit rates for each corresponding position
            cache_hit_rates = []
            ais_cache_hit_rates = []

            for i in range(len(filtered_prefix_hits)):
                if filtered_prefix_queries[i] > 0:
                    cache_hit_rates.append(filtered_prefix_hits[i] / filtered_prefix_queries[i])
                else:
                    cache_hit_rates.append(0.0)

                if filtered_external_queries[i] > 0:
                    ais_cache_hit_rates.append(filtered_external_hits[i] / filtered_external_queries[i])
                else:
                    ais_cache_hit_rates.append(0.0)

            # Store calculated hit rates
            processed_metrics['prefix_cache_hit_rate'] = np.mean(cache_hit_rates) if cache_hit_rates else 0.0
            processed_metrics['external_prefix_cache_hit_rate'] = np.mean(ais_cache_hit_rates) if ais_cache_hit_rates else 0.0
        else:
            # No valid data points
            for key in cache_metric_keys:
                processed_metrics[key] = 0.0
            processed_metrics['prefix_cache_hit_rate'] = 0.0
            processed_metrics['external_prefix_cache_hit_rate'] = 0.0

    # Process lmdeploy metrics normally
    for metric_key, metric_data in llm_metrics.items():
        # Skip cache metrics if they were already processed
        if has_all_cache_metrics and metric_key in cache_metric_keys:
            continue

        if not metric_data:
            processed_metrics[metric_key] = 0.0
            continue

        # Filter out NaN and None values
        valid_data = [x for x in metric_data if not np.isnan(x) and x is not None]

        if not valid_data:
            processed_metrics[metric_key] = 0.0
            continue

        # Remove first 3 and last 2 data points if there are more than 5 data points
        if len(valid_data) > 5:
            filtered_data = valid_data[3:-2]
        else:
            filtered_data = valid_data

        if filtered_data:
            processed_metrics[metric_key] = np.mean(filtered_data)
        else:
            processed_metrics[metric_key] = 0.0

    return processed_metrics
