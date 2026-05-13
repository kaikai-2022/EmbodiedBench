"""
Asset Cache - 资产映射结果本地缓存

为 Normalizer 提供映射结果的持久化缓存，避免重复 LLM 调用。
缓存格式: raw_type → {spec, source_type, is_physical}
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent / "asset_cache.json"


def load_cache() -> Dict[str, Dict]:
    """加载缓存文件，返回 raw_type → {spec, source_type, is_physical} 的映射"""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.debug(f"[Asset Cache] 加载 {len(cache)} 条缓存记录")
        return cache
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"[Asset Cache] 加载失败: {e}，返回空缓存")
        return {}


def save_cache(cache: Dict[str, Dict]) -> None:
    """将缓存写入磁盘"""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.debug(f"[Asset Cache] 写入 {len(cache)} 条缓存记录 → {CACHE_FILE}")
    except IOError as e:
        logger.warning(f"[Asset Cache] 写入失败: {e}")


def get_cached(raw_type: str, cache: Dict) -> Optional[Dict]:
    """从缓存中查找 raw_type 的映射结果"""
    return cache.get(raw_type)


def set_cached(raw_type: str, spec: str, source_type: str, is_physical: bool, cache: Dict) -> None:
    """写入单条缓存记录"""
    cache[raw_type] = {
        "spec": spec,
        "source_type": source_type,
        "is_physical": is_physical,
    }
