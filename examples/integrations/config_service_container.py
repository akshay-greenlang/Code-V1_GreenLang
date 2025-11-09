"""
Integration Example: ConfigManager + ServiceContainer
======================================================

Demonstrates dependency injection and configuration management.
"""

import asyncio
from greenlang.config import ConfigManager, get_config_manager, ServiceContainer


async def main():
    """Run ConfigManager + ServiceContainer integration."""
    print("\nConfigManager + ServiceContainer Integration")
    print("=" * 60)

    # Initialize configuration
    config = get_config_manager()

    # Set configuration values
    config.set("database.url", "postgresql://localhost/greenlang")
    config.set("database.pool_size", 10)
    config.set("cache.enable_l1", True)
    config.set("cache.enable_l2", True)
    config.set("llm.provider", "openai")
    config.set("llm.model", "gpt-4")

    # Initialize service container
    container = ServiceContainer()

    # Register services with dependency injection
    class DatabaseService:
        def __init__(self, config: ConfigManager):
            self.url = config.get("database.url")
            self.pool_size = config.get("database.pool_size")
            print(f"  DatabaseService initialized: {self.url}")

    class CacheService:
        def __init__(self, config: ConfigManager):
            self.enable_l1 = config.get("cache.enable_l1")
            self.enable_l2 = config.get("cache.enable_l2")
            print(f"  CacheService initialized: L1={self.enable_l1}, L2={self.enable_l2}")

    class LLMService:
        def __init__(self, config: ConfigManager):
            self.provider = config.get("llm.provider")
            self.model = config.get("llm.model")
            print(f"  LLMService initialized: {self.provider}/{self.model}")

    class ApplicationService:
        def __init__(self, db: DatabaseService, cache: CacheService, llm: LLMService):
            self.db = db
            self.cache = cache
            self.llm = llm
            print(f"  ApplicationService initialized with all dependencies")

    # Register services in container
    print("\nRegistering services:")
    container.register("config", lambda: config)
    container.register("database", lambda: DatabaseService(config))
    container.register("cache", lambda: CacheService(config))
    container.register("llm", lambda: LLMService(config))
    container.register("app", lambda: ApplicationService(
        db=container.resolve("database"),
        cache=container.resolve("cache"),
        llm=container.resolve("llm")
    ))

    # Resolve services (lazy initialization)
    print("\nResolving services:")
    app = container.resolve("app")

    print(f"\nApplication Service:")
    print(f"  Database: {app.db.url}")
    print(f"  Cache L1: {app.cache.enable_l1}")
    print(f"  LLM: {app.llm.provider}/{app.llm.model}")

    # Configuration can be updated dynamically
    print("\nUpdating configuration:")
    config.set("llm.model", "gpt-3.5-turbo")
    print(f"  New LLM model: {config.get('llm.model')}")

    # Get all configuration
    print("\nAll Configuration:")
    all_config = config.get_all()
    for key, value in all_config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Integration Benefits:")
    print("  - Centralized configuration management")
    print("  - Dependency injection for loose coupling")
    print("  - Lazy service initialization")
    print("  - Easy testing and mocking")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
