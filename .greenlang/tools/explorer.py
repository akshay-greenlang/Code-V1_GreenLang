#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
Interactive Infrastructure Explorer

Web-based interactive explorer using Streamlit.
Browse infrastructure by category, view docs, try examples.
"""

import logging
import argparse
import sys
import os

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    logger.warning(f" streamlit not installed. Run: pip install streamlit")

# Import catalog from infra_search
sys.path.insert(0, os.path.dirname(__file__))
from infra_search import InfrastructureCatalog


def main_app():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GreenLang Infrastructure Explorer",
        page_icon="🌿",
        layout="wide"
    )

    st.title("🌿 GreenLang Infrastructure Explorer")
    st.markdown("Discover and explore GreenLang infrastructure components")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["Browse by Category", "Search", "Dependency Graph", "Quick Start"]
    )

    # Load catalog
    @st.cache_resource
    def load_catalog():
        catalog = InfrastructureCatalog()
        catalog.scan_infrastructure()
        return catalog

    catalog = load_catalog()

    if page == "Browse by Category":
        browse_by_category(catalog)
    elif page == "Search":
        search_page(catalog)
    elif page == "Dependency Graph":
        dependency_graph_page(catalog)
    elif page == "Quick Start":
        quick_start_page()


def browse_by_category(catalog):
    """Browse infrastructure by category."""
    st.header("Browse by Category")

    # Get all categories
    categories = list(set(c.category for c in catalog.components))
    categories.sort()

    # Category selector
    selected_category = st.selectbox("Select a category:", categories)

    # Get components in category
    components = catalog.get_by_category(selected_category)

    st.markdown(f"**{len(components)} components found**")

    # Display components
    for component in components:
        with st.expander(f"📦 {component.name}", expanded=False):
            st.markdown(f"**Category:** {component.category}")
            st.markdown(f"**Description:** {component.description}")

            st.markdown("**When to use:**")
            st.info(component.when_to_use)

            st.markdown("**Example:**")
            st.code(component.code_example, language="python")

            st.markdown(f"**API Methods:** {', '.join(component.api_methods)}")
            st.markdown(f"**Related:** {', '.join(component.related_components)}")
            st.markdown(f"**Tags:** {', '.join(component.tags)}")

            # Try it button
            if st.button(f"Try {component.name}", key=component.name):
                st.code(component.code_example, language="python")
                st.success(f"Copy this code to try {component.name}!")


def search_page(catalog):
    """Search infrastructure."""
    st.header("Search Infrastructure")

    query = st.text_input("Enter your search query:", placeholder="e.g., 'cache API responses'")

    if query:
        results = catalog.search(query, top_k=10)

        st.markdown(f"**{len(results)} results found**")

        for i, component in enumerate(results, 1):
            with st.expander(f"{i}. {component.name}", expanded=i == 1):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{component.description}**")
                    st.markdown("**When to use:**")
                    st.info(component.when_to_use)

                with col2:
                    st.markdown(f"**Category:** {component.category}")
                    st.markdown(f"**Tags:** {', '.join(component.tags[:3])}")

                st.markdown("**Example Code:**")
                st.code(component.code_example, language="python")

                if component.related_components:
                    st.markdown(f"**Related:** {', '.join(component.related_components)}")


def dependency_graph_page(catalog):
    """Show dependency graph."""
    st.header("Infrastructure Dependency Graph")

    st.info("Visual representation of infrastructure dependencies")

    # Simple text-based graph for now
    st.markdown("### Component Relationships")

    for component in catalog.components[:10]:
        st.markdown(f"**{component.name}**")
        if component.related_components:
            for related in component.related_components:
                st.markdown(f"  → {related}")


def quick_start_page():
    """Quick start guide."""
    st.header("Quick Start Guide")

    st.markdown("""
    ## Getting Started with GreenLang Infrastructure

    ### 1. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```

    ### 2. Basic Agent Example

    ```python
    from shared.infrastructure.agents import BaseAgent
    from shared.infrastructure.logging import Logger

    class MyAgent(BaseAgent):
        def __init__(self):
            super().__init__()
            self.logger = Logger(name=__name__)

        def execute(self, input_data):
            self.logger.info("Processing...")
            return {"status": "success"}

    agent = MyAgent()
    result = agent.execute({"data": "test"})
    ```

    ### 3. Using LLM Infrastructure

    ```python
    from shared.infrastructure.llm import ChatSession

    session = ChatSession(provider='openai', model='gpt-4')
    response = session.chat("Analyze this data")
    print(response.content)
    ```

    ### 4. Adding Caching

    ```python
    from shared.infrastructure.cache import CacheManager

    cache = CacheManager(ttl=3600)

    @cache.cached(key_prefix="my_function")
    def expensive_operation(data):
        # Your expensive operation
        return result
    ```

    ### 5. Next Steps

    - Browse infrastructure by category
    - Search for specific functionality
    - Check out example projects
    - Read the full documentation
    """)


def main():
    """CLI entry point."""
    if not HAS_STREAMLIT:
        logger.error(f" streamlit is not installed")
        print("Install with: pip install streamlit")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Launch infrastructure explorer')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on')
    args = parser.parse_args()

    print(f"Starting Infrastructure Explorer on port {args.port}...")
    print(f"Open your browser to: http://localhost:{args.port}")

    # Run streamlit
    os.system(f'streamlit run {__file__} --server.port {args.port}')


if __name__ == '__main__':
    if HAS_STREAMLIT:
        main_app()
    else:
        main()
