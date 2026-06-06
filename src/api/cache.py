"""
Кэширование для ускорения повторяющихся запросов.
Использует LRU-кэш с TTL.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Простой кэш с TTL (Time To Live).
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Инициализация кэша.

        Args:
            max_size: Максимальное количество записей в кэше
            default_ttl: Время жизни записи в секундах (по умолчанию 5 минут)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()

    def _generate_key(self, data: dict) -> str:
        """Генерация ключа кэша на основе данных."""
        # Сортировка ключей для консистентности
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                # TTL истёк
                del self._cache[key]
                return None

            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Сохранение значения в кэш."""
        with self._lock:
            # Очистка устаревших записей если кэш переполнен
            if len(self._cache) >= self._max_size:
                self._cleanup()

            ttl = ttl or self._default_ttl
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl,
                "created_at": time.time(),
            }

    def _cleanup(self):
        """Очистка устаревших записей."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if current_time > entry["expires_at"]
        ]

        for key in expired_keys:
            del self._cache[key]

        # Если всё ещё переполнен, удаляем oldest entries
        if len(self._cache) >= self._max_size:
            sorted_keys = sorted(
                self._cache.keys(), key=lambda k: self._cache[k]["created_at"]
            )
            keys_to_remove = sorted_keys[: len(sorted_keys) // 2]
            for key in keys_to_remove:
                del self._cache[key]

        logger.info(f"Кэш очищен: {len(expired_keys)} устаревших записей")

    def clear(self):
        """Очистка всего кэша."""
        with self._lock:
            self._cache.clear()
            logger.info("Кэш полностью очищен")

    def stats(self) -> Dict[str, int]:
        """Получение статистики кэша."""
        return {"size": len(self._cache), "max_size": self._max_size}


# Глобальный экземпляр кэша
preprocess_cache = SimpleCache(max_size=1000, default_ttl=300)


def cache_prediction(key: str, ttl: int = 300):
    """
    Декоратор для кэширования результатов предсказаний.

    Args:
        key: Ключ кэша (обычно хеш входных данных)
        ttl: Время жизни записи в секундах
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Проверка кэша
            cached = preprocess_cache.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for key: {key[:8]}")
                return cached

            # Выполнение функции
            result = func(*args, **kwargs)

            # Сохранение в кэш
            preprocess_cache.set(key, result, ttl)
            logger.debug(f"Cache miss, result cached for key: {key[:8]}")

            return result

        return wrapper

    return decorator
