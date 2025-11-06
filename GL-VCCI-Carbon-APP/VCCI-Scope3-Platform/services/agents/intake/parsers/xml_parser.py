"""
XML Parser with XPath Support

XML parsing with XPath queries and namespace handling.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..exceptions import FileParseError, XMLPathError
from ..config import get_config

logger = logging.getLogger(__name__)


class XMLParser:
    """
    XML parser with XPath support.

    Features:
    - XPath query support
    - Namespace handling
    - Attribute extraction
    - Nested element parsing
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize XML parser."""
        self.config = get_config().parser if config is None else config
        logger.info("Initialized XMLParser")

    def parse(
        self,
        file_path: Path,
        xpath: Optional[str] = None,
        namespaces: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse XML file into list of dictionaries.

        Args:
            file_path: Path to XML file
            xpath: XPath query to select elements (None = root children)
            namespaces: XML namespaces mapping

        Returns:
            List of dictionaries (one per element)

        Raises:
            FileParseError: If parsing fails
            XMLPathError: If XPath query fails
        """
        try:
            logger.info(f"Parsing XML file: {file_path}")

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Default: get all direct children
            if xpath is None:
                elements = list(root)
            else:
                try:
                    if namespaces:
                        elements = root.findall(xpath, namespaces)
                    else:
                        elements = root.findall(xpath)
                except Exception as e:
                    raise XMLPathError(
                        f"XPath query failed: {str(e)}",
                        details={"xpath": xpath, "error": str(e)}
                    ) from e

            # Convert elements to dictionaries
            records = []
            for elem in elements:
                record = self._element_to_dict(elem, namespaces)
                records.append(record)

            logger.info(f"Successfully parsed {len(records)} records from XML")
            return records

        except (FileParseError, XMLPathError):
            raise

        except ET.ParseError as e:
            raise FileParseError(
                f"Invalid XML: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

        except Exception as e:
            raise FileParseError(
                f"Failed to parse XML file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def _element_to_dict(
        self,
        element: ET.Element,
        namespaces: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert XML element to dictionary.

        Args:
            element: XML element
            namespaces: XML namespaces

        Returns:
            Dictionary representation
        """
        result = {}

        # Add tag name (without namespace)
        tag = element.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        result['_tag'] = tag

        # Add attributes
        if element.attrib:
            result['_attributes'] = dict(element.attrib)

        # Add text content
        if element.text and element.text.strip():
            result['_text'] = element.text.strip()

        # Add child elements
        for child in element:
            child_tag = child.tag
            if '}' in child_tag:
                child_tag = child_tag.split('}', 1)[1]

            child_dict = self._element_to_dict(child, namespaces)

            # If child has no children and no attributes, use text value directly
            if len(child_dict) == 2 and '_tag' in child_dict and '_text' in child_dict:
                child_value = child_dict['_text']
            else:
                child_value = child_dict

            # Handle duplicate child tags
            if child_tag in result:
                # Convert to list
                if not isinstance(result[child_tag], list):
                    result[child_tag] = [result[child_tag]]
                result[child_tag].append(child_value)
            else:
                result[child_tag] = child_value

        return result

    def query_xpath(
        self,
        file_path: Path,
        xpath: str,
        namespaces: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Execute XPath query and return text values.

        Args:
            file_path: Path to XML file
            xpath: XPath query
            namespaces: XML namespaces

        Returns:
            List of text values matching query
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            if namespaces:
                elements = root.findall(xpath, namespaces)
            else:
                elements = root.findall(xpath)

            values = [elem.text.strip() if elem.text else "" for elem in elements]

            logger.info(f"XPath query returned {len(values)} values")
            return values

        except Exception as e:
            raise XMLPathError(
                f"XPath query failed: {str(e)}",
                details={"xpath": xpath, "error": str(e)}
            ) from e

    def extract_with_mapping(
        self,
        file_path: Path,
        xpath_mapping: Dict[str, str],
        namespaces: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract data using XPath mapping.

        Args:
            file_path: Path to XML file
            xpath_mapping: Mapping of field names to XPath queries
                          Example: {"supplier_name": "./name", "country": "./country"}
            namespaces: XML namespaces

        Returns:
            List of dictionaries with mapped fields
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find all record elements (assume root children)
            record_elements = list(root)

            records = []
            for elem in record_elements:
                record = {}
                for field_name, xpath in xpath_mapping.items():
                    try:
                        if namespaces:
                            found = elem.findall(xpath, namespaces)
                        else:
                            found = elem.findall(xpath)

                        if found:
                            if len(found) == 1:
                                record[field_name] = found[0].text if found[0].text else None
                            else:
                                record[field_name] = [
                                    e.text if e.text else None for e in found
                                ]
                    except Exception as e:
                        logger.warning(
                            f"XPath '{xpath}' failed for field '{field_name}': {e}"
                        )
                        record[field_name] = None

                records.append(record)

            logger.info(
                f"Extracted {len(records)} records using XPath mapping"
            )
            return records

        except Exception as e:
            raise FileParseError(
                f"Failed to extract with mapping: {str(e)}",
                details={
                    "file_path": str(file_path),
                    "mapping": xpath_mapping,
                    "error": str(e)
                }
            ) from e

    def detect_namespaces(self, file_path: Path) -> Dict[str, str]:
        """
        Detect XML namespaces in file.

        Args:
            file_path: Path to XML file

        Returns:
            Dictionary of namespace prefixes to URIs
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            namespaces = {}
            for event, elem in ET.iterparse(file_path, events=['start-ns']):
                prefix, uri = elem
                if prefix:
                    namespaces[prefix] = uri
                else:
                    namespaces['default'] = uri

            logger.info(f"Detected namespaces: {namespaces}")
            return namespaces

        except Exception as e:
            logger.warning(f"Failed to detect namespaces: {e}")
            return {}


__all__ = ["XMLParser"]
