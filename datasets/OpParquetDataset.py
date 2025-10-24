import gc
import tempfile
from collections import defaultdict

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from moxing.framework.file import file_io
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from functools import lru_cache
import pickle
import os
from typing import Dict, List, Optional, Any, Callable
import logging
from collections import namedtuple
import multiprocessing

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Python version of Java's String.hashCode()
class JavaHashCode:
    def convert_n_bytes(self, n, b):
        bits = b * 8
        return (n + 2 ** (bits - 1)) % 2 ** bits - 2 ** (bits - 1)

    def convert_4_bytes(self, n):
        return self.convert_n_bytes(n, 4)

    @classmethod
    def getHashCode(cls, s):
        h = 0
        n = len(s)
        for i, c in enumerate(s):
            h = h + ord(c) * 31 ** (n - 1 - i)
        return cls().convert_4_bytes(h)


def _prefetch_worker(batch_id, base_path, cache_dir, feat_config, columns, transform, filter_expr):
    """修复版本的预取工作函数"""
    try:
        base_path_hash = JavaHashCode.getHashCode(base_path)
        cache_key = f"batch_{batch_id}_{base_path_hash}"

        # 检查磁盘缓存是否已存在
        if OptimizedParquetDataset._load_from_disk_cache(cache_key, cache_dir) is not None:
            logger.info(f"Batch {batch_id} already cached, skipping")
            return

        # 创建新的dataset实例避免共享状态
        dataset = ds.dataset(base_path, format="parquet", partitioning="hive")

        # 使用更细粒度的控制
        scanner = dataset.scanner(columns=columns, filter=filter_expr)

        # 直接获取特定批次而不是遍历所有批次
        batch_iter = scanner.to_batches()
        current_batch_id = 0

        for batch in batch_iter:
            if current_batch_id == batch_id:
                # 处理数据
                processed_data = OptimizedParquetDataset._process_batch_data(batch, feat_config, transform)

                # 保存到磁盘缓存
                OptimizedParquetDataset._save_to_disk_cache(cache_key, processed_data, cache_dir)

                # 立即清理当前批次数据
                del processed_data
                del batch
                break

            current_batch_id += 1
            # 显式删除不需要的批次
            del batch

        # 清理资源
        del batch_iter
        del scanner
        del dataset

        # 强制垃圾回收
        gc.collect()

        #logger.info(f"Successfully processed and cached batch {batch_id}")
    except Exception as e:
        logger.error(f"Error in prefetch worker for batch {batch_id}: {e}")
        # 即使出错也要清理内存
        gc.collect()
    finally:
        # 确保所有局部变量都被清理
        locals().clear()

class OptimizedParquetDataset(IterableDataset):
    """
    优化后的Parquet数据集，支持多进程预加载和缓存
    """

    def __init__(self,
                 base_path: str,
                 feat_config: Dict,
                 transform=None,
                 limit: Optional[int] = None,
                 partitions: Optional[Dict] = None,
                 batch_size: int = 100000,
                 prefetch_batches: int = 4,
                 num_workers: int = 16,
                 cache_dir: Optional[str] = None,
                 use_memory_cache: bool = True,
                 memory_cache_size: int = 3,
                 download_data=True):
        """
        Args:
            base_path: Parquet文件目录路径
            feat_config: 特征配置
            transform: 特征转换函数
            limit: 限制读取的总行数
            partitions: 分区过滤条件
            batch_size: 批次大小
            prefetch_batches: 预取批次数量
            num_workers: 工作进程数
            cache_dir: 磁盘缓存目录
            use_memory_cache: 是否使用内存缓存
            memory_cache_size: 内存缓存大小（批次数）
        """
        if download_data:
            # 创建临时目录对象（可 close/cleanup）
            local_temp_obj = tempfile.TemporaryDirectory()
            self.tmp_dir_obj = local_temp_obj  # 保存对象本身，方便后续 cleanup
            file_io.copy_parallel(base_path, local_temp_obj.name)
            logger.info("remote file downloaded finish")
            self.base_path = self.tmp_dir_obj.name
        else:
            self.base_path = base_path
        self.feat_config = feat_config
        self.transform = transform
        self.limit = limit
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.use_memory_cache = use_memory_cache
        self.memory_cache_size = memory_cache_size

        self.columns = [f['name'] for f in feat_config['sparse']] + \
                       [f['name'] for f in feat_config['dense']] + \
                       [f['name'] for f in feat_config['labels']]
        self.hash_gen = JavaHashCode()
        self.base_path_hash = self.hash_gen.getHashCode(self.base_path)
        self.debug_info = {}
        self.cached_files = []
        # 初始化缓存
        if use_memory_cache:
            self.memory_cache = {}
            self.cache_access_order = []

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 设置过滤器
        self.filter_expr = None
        if partitions:
            key = list(partitions.keys())[0]
            self.filter_expr = ds.field(key).isin(partitions[key])

        # 初始化数据集
        self.dataset = ds.dataset(self.base_path, format="parquet", partitioning="hive")

        # 获取所有批次信息
        self._initialize_batches()

        #self.prefetch_executor = ProcessPoolExecutor(max_workers=num_workers)
        #self.prefetch_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_executor = multiprocessing.Pool(processes=num_workers, maxtasksperchild=1)
        self.memory_cache_executor = ThreadPoolExecutor(max_workers=num_workers)

    def _initialize_batches(self):
        """初始化批次信息"""
        scanner = self.dataset.scanner(columns=self.columns, filter=self.filter_expr)
        self.batches_info = []
        total_rows = 0
        for i, batch in enumerate(scanner.to_batches()):
            if self.limit and total_rows >= self.limit:
                break

            batch_rows = len(batch)
            if self.limit and total_rows + batch_rows > self.limit:
                batch_rows = self.limit - total_rows

            self.batches_info.append({
                'batch_id': i,
                'rows': batch_rows,
                'start_row': total_rows
            })
            total_rows += batch_rows
            self.cached_files.append(f"{self.cache_dir}/batch_{i}_{JavaHashCode.getHashCode(self.base_path)}.pkl")

        self.total_rows = total_rows
        logger.info(f"Initialized {len(self.batches_info)} batches with {self.total_rows} total rows")

    def _get_cache_key(self, batch_id: int) -> str:
        """生成缓存键"""
        return f"batch_{batch_id}_{self.base_path_hash}"

    @staticmethod
    def _load_from_disk_cache(cache_key: str, cache_dir: str) -> Optional[Dict]:
        """从磁盘缓存加载数据"""
        if not cache_dir:
            return None

        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk cache {cache_path}: {e}")
        return None

    @staticmethod
    def _save_to_disk_cache(cache_key: str, data: Dict, cache_dir: str):
        """保存数据到磁盘缓存"""
        if not cache_dir:
            return

        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save disk cache {cache_path}: {e}")

    def _get_from_memory_cache(self, cache_key: str) -> Optional[Dict]:
        """从内存缓存获取数据"""
        if not self.use_memory_cache:
            return None

        if cache_key in self.memory_cache:
            # 更新访问顺序
            self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.memory_cache[cache_key]
        return None

    def _put_to_memory_cache(self, cache_key: str, data: Dict):
        """存储数据到内存缓存"""
        if not self.use_memory_cache:
            return

        # 如果缓存已满，移除最老的条目
        while len(self.memory_cache) >= self.memory_cache_size:
            oldest_key = self.cache_access_order.pop(0)
            if oldest_key in self.memory_cache.keys():
                del self.memory_cache[oldest_key]

        self.memory_cache[cache_key] = data
        self.cache_access_order.append(cache_key)

    @staticmethod
    def _batch_hash(string_array: np.ndarray) -> np.ndarray:
        """批量哈希处理字符串数组"""
        # 使用向量化操作加速哈希计算
        hash_values = np.array([JavaHashCode.getHashCode(s) for s in string_array])
        return hash_values

    @staticmethod
    def _batch_transform(df: pd.DataFrame, transform) -> pd.DataFrame:
        """批量应用transform函数"""
        if not transform:
            return df

        return transform(df)

    @staticmethod
    def _batch_to_tensors(df: pd.DataFrame, feat_config) -> Dict:
        """批量转换为PyTorch张量 - 优化内存使用"""
        features = {}
        keys = []

        # 处理稀疏特征
        for feat in feat_config['sparse']:
            name = feat['name']
            if name in df.columns and feat['dtype'] == 'int64':
                keys.append((name, feat['dtype']))
                features[name] = torch.from_numpy(df[name].values.astype(np.int64))

        # 处理密集特征
        for feat in feat_config['dense']:
            name = feat['name']
            if name in df.columns:
                if feat['dtype'] == 'float32':
                    features[name] = torch.from_numpy(df[name].values.astype(np.float32))
                    keys.append((name, feat['dtype']))
                elif feat['dtype'] == 'string':
                    feat_values = df[name].astype(float).values.astype(np.float32)
                    features[name] = torch.from_numpy(feat_values)
                    keys.append((name, feat['dtype']))
        # 处理标签
        for feat in feat_config['labels']:
            name = feat['name']
            if name in df.columns and feat['dtype'] in ['float32', 'int32', 'int64']:
                features[name] = torch.from_numpy(df[name].values.astype(np.float32))
                keys.append((name, feat['dtype']))


        tensors = [features[k[0]].unsqueeze(1) for k in keys]  # 每列变成 (N, 1)
        merged = torch.cat(tensors, dim=1)  # 横向拼接

        return {
            'keys': keys,
            'size': len(df),
            'tensor': merged
        }

    @staticmethod
    def _process_batch_data(batch_data: pa.RecordBatch, feat_config, transform) -> Dict:
        """处理批次数据 - 将所有耗时操作移到这里进行批量处理"""
        df = batch_data.to_pandas()

        # 1. 处理密集特征（批量处理）
        for feat in feat_config['dense']:
            col_name = feat['name']
            if col_name in df.columns:
                default_value = feat.get('default', 0)
                # 转换为数值类型，非数值转为NaN
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                # 处理所有异常情况
                df[col_name] = df[col_name].replace([np.inf, -np.inf, np.nan], default_value)
        # 2. 处理稀疏特征（批量哈希优化）
        for feat in feat_config['sparse']:
            col_name = feat['name']
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna('').astype(str)
                # 批量哈希处理
                df[col_name] = OptimizedParquetDataset._batch_hash(df[col_name].values)

        # 3. 加载labels
        for feat in feat_config['labels']:
            col_name = feat['name']
            if col_name in df.columns:
                default_value = feat.get('default', 0)
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                df[col_name] = df[col_name].replace([np.inf, -np.inf, np.nan], default_value)

        # 4. 批量应用transform函数（如果存在）
        df = OptimizedParquetDataset._batch_transform(df, transform)

        # 5. 批量转换为PyTorch张量
        tensor_dict = OptimizedParquetDataset._batch_to_tensors(df, feat_config)

        # 返回处理后的数据，包含张量和行数信息
        return tensor_dict

    def _disk_to_memory_cache(self, cache_dir, base_path, batch_id: int):
        # 尝试从磁盘缓存加载
        base_path_hash = JavaHashCode.getHashCode(base_path)
        cache_key = f"batch_{batch_id}_{base_path_hash}"
        cached_data = OptimizedParquetDataset._load_from_disk_cache(cache_key, cache_dir)
        if cached_data is not None:
            self._put_to_memory_cache(cache_key, cached_data)
            #logger.info(f"_disk_to_memory_cache {batch_id} success")
            return
        #logger.info(f"_disk_to_memory_cache {batch_id} failed")

    def _load_batch_worker(self, batch_id: int, use_scanner=False) -> Optional[Dict]:
        #logger.info(f"_load_batch_worker ({os.getppid()}) loading batch {batch_id} start")
        """工作进程加载批次数据"""
        cache_key = self._get_cache_key(batch_id)

        # 尝试从内存缓存加载
        if self.use_memory_cache:
            cached_data = self._get_from_memory_cache(cache_key)
            if cached_data is not None:
                #logger.info(f"_load_batch_worker ({os.getppid()}) loading batch {batch_id} memory cached")
                return cached_data

        # 尝试从磁盘缓存加载
        cached_data = OptimizedParquetDataset._load_from_disk_cache(cache_key, self.cache_dir)
        if cached_data is not None:
            if self.use_memory_cache:
                self._put_to_memory_cache(cache_key, cached_data)
            #logger.info(f"_load_batch_worker ({os.getppid()}) loading batch {batch_id} disk cached")
            return cached_data
        if use_scanner:
            # 从原始数据加载
            try:
                scanner = self.dataset.scanner(columns=self.columns, filter=self.filter_expr)
                batches = scanner.to_batches()

                for i, batch in enumerate(batches):
                    if i == batch_id:
                        # 在这里进行所有耗时的批量处理
                        processed_data = OptimizedParquetDataset._process_batch_data(batch, self.feat_config,
                                                                                     self.transform)
                        # 保存到缓存
                        OptimizedParquetDataset._save_to_disk_cache(cache_key, processed_data, self.cache_dir)
                        if self.use_memory_cache:
                            self._put_to_memory_cache(cache_key, processed_data)
                        logger.info(f"_load_batch_worker loading batch {batch_id} finished")
                        return processed_data
                    # 显式释放前一个批次（可选，但有助于大型批次）
                    del batch  # 每个迭代后删除
            except Exception as e:
                logger.error(f"Error loading batch {batch_id}: {e}")
            finally:
                # 无论成功或失败，都清理
                if 'batches' in locals():
                    del batches  # 释放迭代器
                if 'scanner' in locals():
                    del scanner  # 释放scanner
                if 'processed_data' in locals():
                    del processed_data
                if 'batch' in locals():
                    del batch
                gc.collect()  # 手动触发GC，释放Arrow内存
                logger.info(f"Resources cleaned for batch {batch_id}")
        return None

    def _start_prefetching(self, start_batch: int):
        """启动预取"""
        for i in range(self.prefetch_batches):
            batch_id = start_batch + i
            if batch_id < len(self.batches_info):
                self.prefetch_executor.apply_async(
                    _prefetch_worker,
                    (
                        batch_id,
                        self.base_path,
                        self.cache_dir,
                        self.feat_config,
                        self.columns,
                        self.transform,
                        self.filter_expr
                    ),
                    error_callback=lambda e: logger.error(f"Prefetch error for batch {batch_id}: {e}")
                )

    def __iter__(self):
        # 启动预取
        self._start_prefetching(0)
        logger.info('start iter...')
        for batch_info in self.batches_info:
            batch_id = batch_info['batch_id']
            #logger.info(f'inner parquet batch_id : {batch_id}')

            # 获取当前批次的预处理数据（包含张量）
            tensor_dict = None
            wait_times = 0
            while tensor_dict is None and wait_times < 300:
                tensor_dict = self._load_batch_worker(batch_id, use_scanner=False)
                if tensor_dict is None:
                    time.sleep(1)
                    wait_times += 1
            tensor_dict = self._load_batch_worker(batch_id, use_scanner=True)
            if tensor_dict is None:
                continue

            # 预取下一批次
            next_batch_id = batch_id + self.prefetch_batches
            if next_batch_id < len(self.batches_info):
                #logger.info(f'start preloading parquet batch_id : {next_batch_id}')
                self.prefetch_executor.apply_async(
                    _prefetch_worker,
                    (
                        next_batch_id,
                        self.base_path,
                        self.cache_dir,
                        self.feat_config,
                        self.columns,
                        self.transform,
                        self.filter_expr
                    ),
                    error_callback=lambda e: logger.error(f"Prefetch error for batch {batch_id}: {e}")
                )
            # for tensor_dict in tensor_dict:
            #     # # 构建单行特征字典，直接从张量中索引
            #     # features = {}
            #     # for feature_name, tensor_data in tensor_features.items():
            #     #     features[feature_name] = tensor_data[idx]
            #
            #     yield tensor_dict

            # 多线程预取下一批次
            if self.use_memory_cache:
                next_memory_batch_id = batch_id + 1
                self.memory_cache_executor.submit(self._disk_to_memory_cache,
                                                  self.cache_dir,
                                                  self.base_path,
                                                  next_memory_batch_id)
            for idx in range(tensor_dict['size']):
                yield (tensor_dict['keys'], tensor_dict['tensor'][idx].unsqueeze(0))


    def __len__(self):
        return self.total_rows

    def cleanup(self):
        self.tmp_dir_obj.cleanup()
        for path in self.cached_files:
            if os.path.isfile(path):  # 检查是否为文件
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"删除失败 {path} -> {e}")

        """清理资源"""
        if hasattr(self, 'prefetch_executor'):
            try:
                self.prefetch_executor.close()
                self.prefetch_executor.join()
            except Exception as e:
                print(f"prefetch_executor 停止失败 -> {e}")

class BatchedParquetDataset(Dataset):
    """
    批次化的Parquet数据集，适用于常规的DataLoader使用
    """

    def __init__(self,
                 base_path: str,
                 feat_config: Dict,
                 transform=None,
                 limit: Optional[int] = None,
                 partitions: Optional[Dict] = None,
                 batch_size: int = 100000,
                 num_workers: int = 4,
                 cache_dir: Optional[str] = None):
        self.optimized_dataset = OptimizedParquetDataset(
            base_path=base_path,
            feat_config=feat_config,
            transform=transform,
            limit=limit,
            partitions=partitions,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir
        )

        # 预加载所有数据到内存（适用于小到中等规模数据集）
        self.data = list(self.optimized_dataset)
        logger.info(f"Preloaded {len(self.data)} samples into memory")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    start=time.time()
    keys = batch[0][0]  # [(name, dtype), ...]
    batch_tensor = torch.cat([b[1] for b in batch], dim=0)
    features = {}
    for idx, (k, dt) in enumerate(keys):
        t = batch_tensor[:, idx]
        if dt == "int64":
            t = t.long()
        # elif dt in ("float32", "string"):
        #     t = t.float()
        features[k] = t
    end = time.time()
    #logger.info(f"collate_fn cost {end-start} seconds")

    return features

# 使用示例和性能对比函数
def create_optimized_dataloader(base_path: str,
                                feat_config: Dict,
                                batch_size: int = 1024,
                                num_workers: int = 8,
                                prefetch_batches: int = 6,
                                prefetch_factor: int = 2,
                                cache_dir: Optional[str] = None,
                                use_iterable: bool = True,
                                transform: Optional[Callable] = None,
                                is_download_data=True,
                                use_memory_cache=True):
    """
    创建优化后的数据加载器

    Args:
        base_path: 数据路径
        feat_config: 特征配置
        batch_size: DataLoader批次大小
        num_workers: DataLoader工作进程数
        prefetch_batches: 内部batch预加载个数
        prefetch_factor: 预取因子
        cache_dir: 缓存目录
        use_iterable: 是否使用可迭代数据集
        :param transform:
    """

    if use_iterable:
        dataset = OptimizedParquetDataset(
            base_path=base_path,
            feat_config=feat_config,
            cache_dir=cache_dir,
            num_workers=num_workers,
            prefetch_batches = prefetch_batches,
            transform=transform,
            download_data=is_download_data,
            use_memory_cache=use_memory_cache
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # IterableDataset建议使用0
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate_fn
        ), dataset
    else:
        dataset = BatchedParquetDataset(
            base_path=base_path,
            feat_config=feat_config,
            cache_dir=cache_dir,
            num_workers=num_workers
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True  # 支持随机打乱
        ), dataset
