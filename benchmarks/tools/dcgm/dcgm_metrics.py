################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-08-07 17:35:45
# @Details  :
################################################################################
import asyncio
import aiohttp
import sys
import traceback
from prometheus_client.parser import text_string_to_metric_families


desired_metrics = {
    'DCGM_FI_DEV_GPU_UTIL': 'gpu_util',
    'DCGM_FI_PROF_SM_ACTIVE': 'sm_active',
    'DCGM_FI_PROF_SM_OCCUPANCY': 'sm_occupancy',
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE': 'tensor_active',
    'DCGM_FI_DEV_FB_USED': 'gpu_mem',
    'DCGM_FI_PROF_DRAM_ACTIVE': 'dram_active'
}
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)

async def dcgm_metrics(session, device_id, url):
    metrics_dict = {metric: [] for metric in desired_metrics}
    all_dcgm_metrics= {desired_metrics[metric]: 0 for metric in desired_metrics}

    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Error fetching metrics: HTTP {response.status}")
                return all_dcgm_metrics
            metrics_data = await response.text()
    except Exception as e:
        exc_info = sys.exc_info()
        err_msg = "".join(traceback.format_exception(*exc_info))
        print(f"Error fetching metrics: {err_msg}")
        return all_dcgm_metrics

    metrics = text_string_to_metric_families(metrics_data)

    for family in metrics:
        if family.name in desired_metrics:
            for sample in family.samples:
                if sample.labels['gpu'] in device_id:
                    metrics_dict[family.name].append({
                        # 'labels': sample.labels,
                        'value': sample.value
                    })
    print(metrics_dict)
    for metric, data in metrics_dict.items():
        if data:
            average_value = sum(entry['value'] for entry in data) / len(data)
            all_dcgm_metrics[desired_metrics[metric]] = average_value
    return all_dcgm_metrics


async def get_metrics(device_id, fetch_freq_s, stop_event, url='http://localhost:9400/metrics'):
    res_metrics_dict = {}
    res_metrics_dict= {desired_metrics[metric]: [] for metric in desired_metrics}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        while not stop_event.is_set():
            all_dcgm_metrics = await dcgm_metrics(session, str(device_id), url)
            for k, v in all_dcgm_metrics.items():
                res_metrics_dict[k].append(v)
            await asyncio.sleep(fetch_freq_s)
    return res_metrics_dict


async def main():
    device_id="0 1 2 3"
    fetch_freq_s=1
    stop_event = asyncio.Event()

    gpu_metrics_task = asyncio.create_task(get_metrics(device_id, fetch_freq_s, stop_event))

    try:
        await asyncio.sleep(10)
    finally:
        stop_event.set()
        res_metrics_dict = await gpu_metrics_task

    print(res_metrics_dict)

if __name__ == "__main__":
    asyncio.run(main())
