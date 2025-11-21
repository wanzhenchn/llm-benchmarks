import asyncio
import aiohttp
import sys
import traceback
import logging
from typing import Dict, List, Optional
import importlib.util

from prometheus_client.parser import text_string_to_metric_families


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)


class LLMMetricsMonitor:
    """vLLM 监控指标收集器"""

    def __init__(self,
                 metrics_url: str = 'http://localhost:8000/metrics',
                 backend: str = 'vllm'):
        # 定义要提取的 LLM 指标名称
        if backend.lower() == 'vllm':
            METRICS = {
                'vllm:prefix_cache_hits': 'prefix_cache_hits',
                'vllm:prefix_cache_queries': 'prefix_cache_queries',
                'vllm:external_prefix_cache_hits': 'external_prefix_cache_hits',
                'vllm:external_prefix_cache_queries': 'external_prefix_cache_queries'
            }
        elif backend.lower() == 'lmdeploy':
            METRICS = {
                'lmdeploy:gpu_cache_usage_perc': 'gpu_cache_usage_perc'
            }
        else:
            raise ValueError(f'backend type only supports vllm and lmdeploy.')
        self.METRICS =  METRICS

        self.metrics_url = metrics_url
        self.metrics_data = {metric: [] for metric in self.METRICS.values()}

    async def fetch_metrics(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """获取单次监控指标"""
        current_metrics = {metric: 0.0 for metric in self.METRICS.values()}

        try:
            async with session.get(self.metrics_url) as response:
                if response.status != 200:
                    logging.warning(f"Error fetching vLLM metrics: HTTP {response.status}")
                    return current_metrics

                metrics_data = await response.text()

        except Exception as e:
            exc_info = sys.exc_info()
            err_msg = "".join(traceback.format_exception(*exc_info))
            logging.error(f"Error fetching vLLM metrics: {err_msg}")
            return current_metrics

        # 解析 Prometheus 格式的指标
        try:
            metrics = text_string_to_metric_families(metrics_data)

            for family in metrics:
                if family.name in self.METRICS:
                    metric_key = self.METRICS[family.name]
                    logging.debug(f"Processing metric: {family.name} -> {metric_key}")

                    # 处理不同类型的指标
                    if family.type == 'gauge' or family.type == 'counter':
                        # 对于 gauge 和 counter 类型，取所有样本的平均值
                        values = [sample.value for sample in family.samples]
                        logging.debug(f"Sample values: {values}")

                        if values:
                            # 过滤掉 NaN 和 None 值
                            valid_values = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]

                            if valid_values:
                                avg_value = sum(valid_values) / len(valid_values)
                                current_metrics[metric_key] = avg_value
                                logging.debug(f"Calculated average: {avg_value}")
                            else:
                                logging.warning(f"No valid values found for {family.name}")
                                current_metrics[metric_key] = 0.0
                        else:
                            logging.warning(f"No samples found for {family.name}")
                            current_metrics[metric_key] = 0.0

                    elif family.type == 'histogram':
                        # 对于 histogram 类型，计算平均值
                        total_sum = 0
                        total_count = 0
                        for sample in family.samples:
                            if sample.name.endswith('_sum'):
                                total_sum += sample.value
                            elif sample.name.endswith('_count'):
                                total_count += sample.value

                        if total_count > 0:
                            current_metrics[metric_key] = total_sum / total_count
                        else:
                            current_metrics[metric_key] = 0.0

        except Exception as e:
            logging.error(f"Error parsing vLLM metrics: {e}")
            import traceback
            logging.error(traceback.format_exc())

        return current_metrics

    async def collect_metrics_periodically(self,
                                         fetch_freq_s: int,
                                         stop_event: asyncio.Event) -> Dict[str, List[float]]:
        """定期收集监控指标"""
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            while not stop_event.is_set():
                try:
                    current_metrics = await self.fetch_metrics(session)

                    # 存储指标数据
                    for metric_name, value in current_metrics.items():
                        self.metrics_data[metric_name].append(value)

                    logging.debug(f"Collected vLLM metrics: {current_metrics}")

                except Exception as e:
                    logging.error(f"Error in periodic metrics collection: {e}")

                await asyncio.sleep(fetch_freq_s)

        return self.metrics_data

    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标值"""
        avg_metrics = {}
        for metric_name, values in self.metrics_data.items():
            if values:
                # 排除前3个和后2个数据点（类似 GPU 指标的处理方式）
                if len(values) > 5:
                    filtered_values = values[3:-2]
                else:
                    filtered_values = values
                avg_metrics[metric_name] = sum(filtered_values) / len(filtered_values) if filtered_values else 0.0
            else:
                avg_metrics[metric_name] = 0.0

        return avg_metrics

    def get_max_metrics(self) -> Dict[str, float]:
        """获取最大指标值"""
        max_metrics = {}
        for metric_name, values in self.metrics_data.items():
            if values:
                max_metrics[metric_name] = max(values)
            else:
                max_metrics[metric_name] = 0.0

        return max_metrics

    def get_min_metrics(self) -> Dict[str, float]:
        """获取最小指标值"""
        min_metrics = {}
        for metric_name, values in self.metrics_data.items():
            if values:
                min_metrics[metric_name] = min(values)
            else:
                min_metrics[metric_name] = 0.0

        return min_metrics


async def get_llm_metrics_periodically(metrics_url: str,
                                       fetch_freq_s: int,
                                       stop_event: asyncio.Event,
                                       backend: str) -> Dict[str, List[float]]:
    monitor = LLMMetricsMonitor(metrics_url, backend)
    return await monitor.collect_metrics_periodically(fetch_freq_s, stop_event)


async def main():
    """测试函数"""
    fetch_freq_s = 1
    stop_event = asyncio.Event()

    # 测试 vLLM 监控
    metrics_task = asyncio.create_task(
        get_llm_metrics_periodically(
            'http://localhost:8000/metrics', fetch_freq_s, stop_event, 'vllm'
        )
    )

    try:
        await asyncio.sleep(10)
    finally:
        stop_event.set()
        metrics = await metrics_task

    print("Metrics collected:")
    for metric, values in metrics.items():
        if values:
            print(f"{metric}: {len(values)} samples, avg: {sum(values)/len(values):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
