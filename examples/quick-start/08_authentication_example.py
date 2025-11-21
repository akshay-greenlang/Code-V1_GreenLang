# -*- coding: utf-8 -*-
"""
Example 8: Authentication Integration
======================================

Demonstrates authentication and authorization with GreenLang.
"""

import asyncio
from greenlang.auth import AuthManager, User, Role, Permission
from greenlang.security import SecurityContext


async def main():
    """Run authentication example."""
    # Initialize auth manager
    auth = AuthManager()

    # Create roles
    admin_role = Role(
        name="admin",
        permissions=[
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.ADMIN
        ]
    )

    analyst_role = Role(
        name="analyst",
        permissions=[
            Permission.READ,
            Permission.WRITE
        ]
    )

    # Create users
    admin_user = User(
        id="user1",
        username="admin@example.com",
        roles=[admin_role]
    )

    analyst_user = User(
        id="user2",
        username="analyst@example.com",
        roles=[analyst_role]
    )

    # Authenticate user
    print("Authentication Examples:")

    # Admin user operations
    print(f"\nAdmin user ({admin_user.username}):")
    print(f"  Can read: {auth.has_permission(admin_user, Permission.READ)}")
    print(f"  Can write: {auth.has_permission(admin_user, Permission.WRITE)}")
    print(f"  Can delete: {auth.has_permission(admin_user, Permission.DELETE)}")
    print(f"  Can admin: {auth.has_permission(admin_user, Permission.ADMIN)}")

    # Analyst user operations
    print(f"\nAnalyst user ({analyst_user.username}):")
    print(f"  Can read: {auth.has_permission(analyst_user, Permission.READ)}")
    print(f"  Can write: {auth.has_permission(analyst_user, Permission.WRITE)}")
    print(f"  Can delete: {auth.has_permission(analyst_user, Permission.DELETE)}")
    print(f"  Can admin: {auth.has_permission(analyst_user, Permission.ADMIN)}")

    # Security context
    with SecurityContext(user=admin_user):
        print(f"\nOperating in security context as: {admin_user.username}")
        # Perform authorized operations


if __name__ == "__main__":
    asyncio.run(main())
