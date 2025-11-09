# Workshop 4: Data Management & Caching Strategies

**Duration:** 2 hours
**Level:** Intermediate
**Prerequisites:** Workshops 1-3 completed

---

## Workshop Overview

Master data management using GreenLang's caching, validation, and database infrastructure. Learn to build fast, reliable applications with proper data handling.

### Learning Objectives

- Use CacheManager for Redis caching
- Implement ValidationFramework for data quality
- Use DatabaseManager abstraction
- Design caching strategies
- Handle cache invalidation
- Optimize database queries

---

## Part 1: CacheManager (30 minutes)

### Why Caching?

**Performance:**
- Database query: 50-200ms
- Cache lookup: 1-5ms
- **40x faster!**

**Cost:**
- LLM API call: $0.01
- Cache hit: $0.00
- **100% savings!**

### CacheManager Basics

```python
from GL_COMMONS.infrastructure.cache import CacheManager

# Initialize
cache = CacheManager()

# Set value (expires in 1 hour)
cache.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=3600)

# Get value
user = cache.get("user:123")
print(user)  # {"name": "John", "email": "john@example.com"}

# Delete
cache.delete("user:123")

# Check existence
exists = cache.exists("user:123")  # False
```

### TTL (Time To Live)

```python
# Short-lived (5 minutes) - frequently changing data
cache.set("stock_price", 150.25, ttl=300)

# Medium-lived (1 hour) - semi-static data
cache.set("user_profile", profile_data, ttl=3600)

# Long-lived (24 hours) - rarely changing data
cache.set("company_info", company_data, ttl=86400)

# Permanent (no expiration) - static data
cache.set("emission_factors", factors, ttl=None)
```

### Cache Patterns

#### 1. Cache-Aside (Lazy Loading)

```python
def get_user(user_id):
    """Cache-aside pattern."""

    # Try cache first
    cache_key = f"user:{user_id}"
    user = cache.get(cache_key)

    if user:
        logger.info("Cache hit")
        return user

    # Cache miss - load from database
    logger.info("Cache miss - loading from DB")
    user = db.query("SELECT * FROM users WHERE id=?", [user_id])

    # Store in cache
    cache.set(cache_key, user, ttl=3600)

    return user
```

#### 2. Write-Through

```python
def update_user(user_id, data):
    """Write-through pattern."""

    # Update database
    db.execute("UPDATE users SET ... WHERE id=?", [user_id, ...])

    # Update cache immediately
    cache_key = f"user:{user_id}"
    cache.set(cache_key, data, ttl=3600)

    return data
```

#### 3. Write-Behind (Write-Back)

```python
def update_user_async(user_id, data):
    """Write-behind pattern."""

    # Update cache immediately
    cache_key = f"user:{user_id}"
    cache.set(cache_key, data, ttl=3600)

    # Queue database update for later
    task_queue.enqueue("update_user_in_db", user_id, data)

    return data
```

### Advanced Caching

#### Batch Operations

```python
# Set multiple keys at once
data = {
    "user:1": {"name": "Alice"},
    "user:2": {"name": "Bob"},
    "user:3": {"name": "Charlie"}
}
cache.set_many(data, ttl=3600)

# Get multiple keys
keys = ["user:1", "user:2", "user:3"]
users = cache.get_many(keys)
```

#### Pattern-Based Deletion

```python
# Delete all user caches
cache.delete_pattern("user:*")

# Delete all emission caches for 2023
cache.delete_pattern("emission:2023:*")

# Delete all temporary caches
cache.delete_pattern("temp:*")
```

#### Cache Warming

```python
def warm_cache():
    """Pre-load frequently accessed data."""

    # Load top 100 companies
    companies = db.query("SELECT * FROM companies ORDER BY access_count DESC LIMIT 100")

    for company in companies:
        cache.set(f"company:{company['id']}", company, ttl=86400)

    logger.info("Cache warmed with 100 companies")
```

---

## Part 2: ValidationFramework (25 minutes)

### Why Validation?

**Data Quality:**
- Prevent garbage in, garbage out
- Catch errors early
- Ensure consistency

**Security:**
- Prevent injection attacks
- Validate user input
- Sanitize data

### ValidationFramework Basics

```python
from GL_COMMONS.infrastructure.validation import ValidationFramework

validator = ValidationFramework()

# Define schema
schema = {
    "name": {"type": "string", "required": True, "min_length": 2, "max_length": 100},
    "email": {"type": "email", "required": True},
    "age": {"type": "integer", "min": 0, "max": 150},
    "role": {"type": "string", "enum": ["admin", "user", "guest"]}
}

# Validate data
data = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30,
    "role": "admin"
}

# Raises exception if invalid
validator.validate(data, schema)
```

### Validation Types

#### String Validation

```python
schema = {
    "company_name": {
        "type": "string",
        "required": True,
        "min_length": 2,
        "max_length": 200,
        "pattern": r"^[A-Za-z0-9\s]+$"  # Alphanumeric only
    },
    "url": {
        "type": "url",
        "required": False
    },
    "email": {
        "type": "email",
        "required": True
    }
}
```

#### Number Validation

```python
schema = {
    "emissions": {
        "type": "number",
        "required": True,
        "min": 0,
        "max": 1000000
    },
    "year": {
        "type": "integer",
        "required": True,
        "min": 2020,
        "max": 2030
    },
    "percentage": {
        "type": "number",
        "min": 0,
        "max": 100
    }
}
```

#### Nested Object Validation

```python
schema = {
    "company": {
        "type": "object",
        "required": True,
        "schema": {
            "name": {"type": "string", "required": True},
            "id": {"type": "string", "required": True}
        }
    },
    "emissions": {
        "type": "object",
        "required": True,
        "schema": {
            "scope_1": {"type": "number", "min": 0},
            "scope_2": {"type": "number", "min": 0},
            "scope_3": {"type": "number", "min": 0}
        }
    }
}

data = {
    "company": {
        "name": "Tesla",
        "id": "TSLA"
    },
    "emissions": {
        "scope_1": 100,
        "scope_2": 200,
        "scope_3": 300
    }
}

validator.validate(data, schema)
```

#### Array Validation

```python
schema = {
    "companies": {
        "type": "array",
        "required": True,
        "min_items": 1,
        "max_items": 100,
        "item_schema": {
            "type": "object",
            "schema": {
                "name": {"type": "string", "required": True},
                "emissions": {"type": "number", "min": 0}
            }
        }
    }
}
```

### Custom Validators

```python
def validate_emission_year(value):
    """Custom validator for emission year."""
    current_year = datetime.now().year
    if value > current_year:
        raise ValueError(f"Year cannot be in the future: {value}")
    if value < 1990:
        raise ValueError(f"Year too old: {value}")
    return True

# Register custom validator
validator.register_custom_validator("emission_year", validate_emission_year)

# Use in schema
schema = {
    "year": {
        "type": "integer",
        "custom_validator": "emission_year"
    }
}
```

---

## Part 3: DatabaseManager (25 minutes)

### DatabaseManager Abstraction

```python
from GL_COMMONS.infrastructure.database import DatabaseManager

# Initialize (connection from config)
db = DatabaseManager()

# Simple query
results = db.query("SELECT * FROM companies WHERE year = ?", [2023])

# Single row
company = db.query_one("SELECT * FROM companies WHERE id = ?", ["TSLA"])

# Execute (INSERT, UPDATE, DELETE)
db.execute(
    "INSERT INTO emissions (company, year, scope_1) VALUES (?, ?, ?)",
    ["Tesla", 2023, 100]
)

# Bulk insert
data = [
    {"company": "Tesla", "year": 2023, "scope_1": 100},
    {"company": "Apple", "year": 2023, "scope_1": 150}
]
db.bulk_insert("emissions", data)
```

### Transaction Support

```python
# Atomic transaction
with db.transaction():
    # All or nothing
    db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    db.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")

# If any fails, all rollback
```

### Query Builder

```python
# Build complex queries
query = (
    db.query_builder()
    .select("companies.name", "emissions.scope_1", "emissions.scope_2")
    .from_table("companies")
    .join("emissions", "companies.id = emissions.company_id")
    .where("emissions.year = ?", [2023])
    .where("emissions.scope_1 > ?", [100])
    .order_by("emissions.scope_1", "DESC")
    .limit(10)
)

results = query.execute()
```

### Caching Database Queries

```python
def get_company_emissions(company_id, year):
    """Get emissions with caching."""

    cache_key = f"emissions:{company_id}:{year}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Query database
    results = db.query(
        "SELECT * FROM emissions WHERE company_id = ? AND year = ?",
        [company_id, year]
    )

    # Cache for 1 hour
    cache.set(cache_key, results, ttl=3600)

    return results
```

---

## Part 4: Hands-On Lab - Caching Strategy (40 minutes)

### Lab: Build a Multi-Layer Cache

**Requirements:**
1. Three cache layers: L1 (memory), L2 (Redis), L3 (database)
2. Automatic cache warming
3. Cache invalidation on updates
4. Performance tracking

### Implementation

```python
# multi_layer_cache.py
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.database import DatabaseManager
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class MultiLayerCache:
    """Three-layer caching system."""

    def __init__(self):
        self.l1_cache = {}  # In-memory (fastest)
        self.l2_cache = CacheManager()  # Redis (fast)
        self.db = DatabaseManager()  # Database (slowest)

        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0
        }

    def get(self, key):
        """Get value from cache (L1 -> L2 -> L3)."""

        # Try L1 (memory)
        if key in self.l1_cache:
            self.stats["l1_hits"] += 1
            logger.debug(f"L1 hit: {key}")
            return self.l1_cache[key]

        # Try L2 (Redis)
        value = self.l2_cache.get(key)
        if value:
            self.stats["l2_hits"] += 1
            logger.debug(f"L2 hit: {key}")
            # Promote to L1
            self.l1_cache[key] = value
            return value

        # Try L3 (Database)
        value = self._load_from_database(key)
        if value:
            self.stats["l3_hits"] += 1
            logger.debug(f"L3 hit: {key}")
            # Promote to L2 and L1
            self.l2_cache.set(key, value, ttl=3600)
            self.l1_cache[key] = value
            return value

        # Miss
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key, value, ttl=3600):
        """Set value in all cache layers."""

        # Set in all layers
        self.l1_cache[key] = value
        self.l2_cache.set(key, value, ttl=ttl)
        self._save_to_database(key, value)

        logger.debug(f"Cached: {key}")

    def invalidate(self, key):
        """Invalidate cache entry."""

        # Remove from all layers
        if key in self.l1_cache:
            del self.l1_cache[key]

        self.l2_cache.delete(key)

        logger.debug(f"Invalidated: {key}")

    def get_stats(self):
        """Get cache statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "l1_hit_rate": (self.stats["l1_hits"] / total) * 100,
            "l2_hit_rate": (self.stats["l2_hits"] / total) * 100,
            "l3_hit_rate": (self.stats["l3_hits"] / total) * 100,
            "miss_rate": (self.stats["misses"] / total) * 100
        }

    def _load_from_database(self, key):
        """Load value from database."""
        # Parse key: "emission:company_id:year"
        parts = key.split(":")
        if len(parts) != 3 or parts[0] != "emission":
            return None

        company_id = parts[1]
        year = parts[2]

        result = self.db.query_one(
            "SELECT * FROM emissions WHERE company_id = ? AND year = ?",
            [company_id, year]
        )

        return result

    def _save_to_database(self, key, value):
        """Save value to database."""
        # In real implementation, would update database
        pass

# Usage Example
cache = MultiLayerCache()

# First access - loads from database (L3)
data = cache.get("emission:TSLA:2023")  # L3 hit

# Second access - from Redis (L2)
data = cache.get("emission:TSLA:2023")  # L2 hit

# Third access - from memory (L1)
data = cache.get("emission:TSLA:2023")  # L1 hit (fastest!)

# Check stats
stats = cache.get_stats()
print(f"L1 hit rate: {stats['l1_hit_rate']:.1f}%")
print(f"L2 hit rate: {stats['l2_hit_rate']:.1f}%")
print(f"L3 hit rate: {stats['l3_hit_rate']:.1f}%")
```

### Cache Warming

```python
def warm_cache():
    """Pre-load frequently accessed data."""

    # Get top 100 most accessed companies
    companies = db.query("""
        SELECT company_id, year, COUNT(*) as access_count
        FROM access_log
        GROUP BY company_id, year
        ORDER BY access_count DESC
        LIMIT 100
    """)

    for company in companies:
        key = f"emission:{company['company_id']}:{company['year']}"
        data = db.query_one(
            "SELECT * FROM emissions WHERE company_id = ? AND year = ?",
            [company['company_id'], company['year']]
        )
        cache.set(key, data)

    logger.info(f"Warmed cache with {len(companies)} entries")
```

---

## Part 5: Cache Invalidation Strategies (15 minutes)

### Time-Based Invalidation

```python
# Short TTL for frequently changing data
cache.set("stock_price:TSLA", 150.25, ttl=60)  # 1 minute

# Long TTL for static data
cache.set("company_info:TSLA", company_data, ttl=86400)  # 24 hours
```

### Event-Based Invalidation

```python
def update_emission(company_id, year, data):
    """Update emission and invalidate cache."""

    # Update database
    db.execute(
        "UPDATE emissions SET scope_1=?, scope_2=?, scope_3=? WHERE company_id=? AND year=?",
        [data["scope_1"], data["scope_2"], data["scope_3"], company_id, year]
    )

    # Invalidate related caches
    cache.delete(f"emission:{company_id}:{year}")
    cache.delete(f"total_emission:{company_id}")
    cache.delete(f"yearly_report:{year}")

    logger.info(f"Invalidated caches for {company_id}:{year}")
```

### Tag-Based Invalidation

```python
# Tag caches for group invalidation
cache.set_with_tags(
    key="emission:TSLA:2023",
    value=data,
    tags=["company:TSLA", "year:2023", "scope:all"]
)

# Invalidate all caches for TSLA
cache.invalidate_by_tag("company:TSLA")

# Invalidate all caches for 2023
cache.invalidate_by_tag("year:2023")
```

---

## Workshop Wrap-Up

### What You Learned

✓ CacheManager for Redis caching
✓ ValidationFramework for data quality
✓ DatabaseManager abstraction
✓ Multi-layer caching strategies
✓ Cache invalidation patterns
✓ Performance optimization

### Key Takeaways

1. **Always cache** - 40x faster than database
2. **Validate early** - Catch errors at the edge
3. **Use abstractions** - Never use psycopg2 directly
4. **Invalidate carefully** - Stale cache is worse than no cache
5. **Monitor performance** - Track hit rates and costs

### Homework

Build a caching system:
1. Implement multi-layer cache
2. Add cache warming
3. Implement invalidation strategy
4. Track and optimize hit rates
5. Measure performance improvement

---

**Workshop Complete! Ready for Workshop 5: Monitoring & Production**
