"""
Centralized Cache Management System
Provides unified caching with TTL support and statistics
"""

import hashlib
import json
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
from custom_logging.my_logger import logger
from exception.exceptions import CacheException
import sys


class CacheManager:
    """
    Centralized cache manager with TTL support
    Supports multiple cache types: response cache, query cache, session cache
    """
    
    def __init__(self, default_ttl_seconds: int = 1800):
        """
        Initialize cache manager
        
        Args:
            default_ttl_seconds: Default TTL for cached items (30 minutes)
        """
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl_seconds
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        logger.info(f"[CACHE] Initialized with TTL: {default_ttl_seconds}s")
    
    def _generate_key(self, *args, prefix: str = "", **kwargs) -> str:
        """
        Generate cache key from arguments
        
        Args:
            args: Positional arguments
            prefix: Key prefix for namespacing
            kwargs: Keyword arguments
        
        Returns:
            MD5 hash of serialized arguments
        """
        try:
            # Serialize arguments
            key_data = {
                'args': [str(arg) for arg in args],
                'kwargs': {k: str(v) for k, v in kwargs.items()}
            }
            key_string = json.dumps(key_data, sort_keys=True)
            
            # Generate hash
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            
            return f"{prefix}:{key_hash}" if prefix else key_hash
            
        except Exception as e:
            logger.error(f"[CACHE] Key generation failed: {e}")
            raise CacheException(f"Failed to generate cache key: {e}", sys)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check if expired
                if datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                    self.stats['hits'] += 1
                    logger.debug(f"[CACHE HIT] Key: {key[:20]}...")
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.stats['evictions'] += 1
                    logger.debug(f"[CACHE EXPIRED] Key: {key[:20]}...")
            
            self.stats['misses'] += 1
            logger.debug(f"[CACHE MISS] Key: {key[:20]}...")
            return None
            
        except Exception as e:
            logger.error(f"[CACHE] Get failed for key {key[:20]}: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL (overrides default)
        
        Returns:
            True if successful
        """
        try:
            self.cache[key] = (value, datetime.now())
            self.stats['sets'] += 1
            
            ttl = ttl_seconds or self.default_ttl
            logger.debug(f"[CACHE SET] Key: {key[:20]}... | TTL: {ttl}s")
            
            return True
            
        except Exception as e:
            logger.error(f"[CACHE] Set failed for key {key[:20]}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"[CACHE DELETE] Key: {key[:20]}...")
                return True
            return False
        except Exception as e:
            logger.error(f"[CACHE] Delete failed for key {key[:20]}: {e}")
            return False
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            prefix: Clear only keys with this prefix (None = clear all)
        
        Returns:
            Number of entries cleared
        """
        try:
            if prefix:
                keys_to_delete = [k for k in self.cache.keys() if k.startswith(prefix)]
                for key in keys_to_delete:
                    del self.cache[key]
                count = len(keys_to_delete)
                logger.info(f"[CACHE CLEAR] Cleared {count} entries with prefix '{prefix}'")
            else:
                count = len(self.cache)
                self.cache.clear()
                logger.info(f"[CACHE CLEAR] Cleared all {count} entries")
            
            return count
            
        except Exception as e:
            logger.error(f"[CACHE] Clear failed: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries
        
        Returns:
            Number of entries removed
        """
        try:
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if now - timestamp >= timedelta(seconds=self.default_ttl)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            count = len(expired_keys)
            self.stats['evictions'] += count
            
            if count > 0:
                logger.info(f"[CACHE CLEANUP] Removed {count} expired entries")
            
            return count
            
        except Exception as e:
            logger.error(f"[CACHE] Cleanup failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        logger.info("[CACHE] Statistics reset")


# Singleton instance
_cache_manager_instance = None


def get_cache_manager(ttl_seconds: int = 1800) -> CacheManager:
    """Get or create cache manager singleton"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager(default_ttl_seconds=ttl_seconds)
    
    return _cache_manager_instance


def cached(
    prefix: str = "",
    ttl_seconds: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Args:
        prefix: Cache key prefix
        ttl_seconds: Custom TTL
        key_builder: Custom function to build cache key
    
    Example:
        @cached(prefix="document", ttl_seconds=3600)
        def expensive_query(doc_id: str) -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_mgr = get_cache_manager()
            
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache_mgr._generate_key(*args, prefix=prefix, **kwargs)
            
            # Try to get from cache
            cached_value = cache_mgr.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_mgr.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def async_cached(
    prefix: str = "",
    ttl_seconds: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Async decorator for caching function results
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_mgr = get_cache_manager()
            
            # Generate cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache_mgr._generate_key(*args, prefix=prefix, **kwargs)
            
            # Try to get from cache
            cached_value = cache_mgr.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache_mgr.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


# Export
__all__ = [
    'CacheManager',
    'get_cache_manager',
    'cached',
    'async_cached'
]
