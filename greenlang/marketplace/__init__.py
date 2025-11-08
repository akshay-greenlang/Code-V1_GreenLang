"""
GreenLang Agent Marketplace

The marketplace module provides a comprehensive platform for publishing,
discovering, and monetizing GreenLang agents. It includes:

- Agent publishing and versioning
- Ratings and reviews
- Search and discovery
- Dependency management
- Monetization and licensing
- Recommendation engine
"""

from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentVersion,
    AgentReview,
    AgentCategory,
    AgentTag,
    AgentAsset,
    AgentDependency,
    AgentLicense,
    AgentInstall,
    AgentPurchase,
)

from greenlang.marketplace.rating_system import (
    RatingSystem,
    ReviewModerator,
    calculate_wilson_score,
)

from greenlang.marketplace.recommendation import (
    RecommendationEngine,
    CollaborativeFilter,
    ContentBasedFilter,
)

from greenlang.marketplace.publisher import (
    AgentPublisher,
    PublishingWorkflow,
    ValidationResult,
)

from greenlang.marketplace.validator import (
    AgentValidator,
    CodeValidator,
    SecurityScanner,
)

from greenlang.marketplace.versioning import (
    VersionManager,
    SemanticVersion,
    BreakingChangeDetector,
)

from greenlang.marketplace.dependency_resolver import (
    DependencyResolver,
    DependencyGraph,
    VersionConflictResolver,
)

from greenlang.marketplace.search import (
    AgentSearchEngine,
    SearchFilter,
    SearchSuggestions,
)

from greenlang.marketplace.categories import (
    CategoryManager,
    CATEGORY_HIERARCHY,
    get_category_tree,
)

from greenlang.marketplace.monetization import (
    MonetizationManager,
    PaymentProcessor,
    PricingModel,
)

from greenlang.marketplace.license_manager import (
    LicenseManager,
    LicenseKey,
    LicenseValidator,
)

__all__ = [
    # Models
    "MarketplaceAgent",
    "AgentVersion",
    "AgentReview",
    "AgentCategory",
    "AgentTag",
    "AgentAsset",
    "AgentDependency",
    "AgentLicense",
    "AgentInstall",
    "AgentPurchase",
    # Rating System
    "RatingSystem",
    "ReviewModerator",
    "calculate_wilson_score",
    # Recommendation
    "RecommendationEngine",
    "CollaborativeFilter",
    "ContentBasedFilter",
    # Publishing
    "AgentPublisher",
    "PublishingWorkflow",
    "ValidationResult",
    # Validation
    "AgentValidator",
    "CodeValidator",
    "SecurityScanner",
    # Versioning
    "VersionManager",
    "SemanticVersion",
    "BreakingChangeDetector",
    # Dependencies
    "DependencyResolver",
    "DependencyGraph",
    "VersionConflictResolver",
    # Search
    "AgentSearchEngine",
    "SearchFilter",
    "SearchSuggestions",
    # Categories
    "CategoryManager",
    "CATEGORY_HIERARCHY",
    "get_category_tree",
    # Monetization
    "MonetizationManager",
    "PaymentProcessor",
    "PricingModel",
    # Licensing
    "LicenseManager",
    "LicenseKey",
    "LicenseValidator",
]
