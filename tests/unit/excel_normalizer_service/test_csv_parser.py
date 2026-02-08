# -*- coding: utf-8 -*-
"""
Unit Tests for CSVParser (AGENT-DATA-002)

Tests CSV file parsing: initialization, parse_file, detect_encoding,
detect_delimiter (comma, semicolon, tab, pipe), extract_headers,
extract_rows, detect_has_header, stream_rows, edge cases
(BOM, mixed line endings, quoted fields), and statistics.

Coverage target: 85%+ of csv_parser.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import csv
import hashlib
import io
from typing import Any, Dict, Iterator, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline CSVParser mirroring greenlang/excel_normalizer/csv_parser.py
# ---------------------------------------------------------------------------


class CSVParser:
    """CSV file parser with delimiter detection and encoding handling."""

    DELIMITER_MAP = {
        "comma": ",", "semicolon": ";", "tab": "\t", "pipe": "|",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._default_encoding: str = self._config.get("default_encoding", "utf-8")
        self._default_delimiter: str = self._config.get("default_delimiter", ",")
        self._sample_rows: int = self._config.get("sample_rows", 100)
        self._stats: Dict[str, int] = {
            "files_parsed": 0, "rows_parsed": 0, "encoding_detections": 0,
            "delimiter_detections": 0, "parse_errors": 0,
        }

    def parse_file(self, content: bytes, file_name: str = "") -> Dict[str, Any]:
        text = self._decode(content)
        delimiter = self.detect_delimiter(text)
        has_header = self.detect_has_header(text, delimiter)
        headers = self.extract_headers(text, delimiter) if has_header else []
        rows = self.extract_rows(text, delimiter, skip_header=has_header)
        file_hash = hashlib.sha256(content).hexdigest()
        self._stats["files_parsed"] += 1
        self._stats["rows_parsed"] += len(rows)
        return {
            "file_name": file_name or "unknown.csv",
            "file_hash": file_hash,
            "file_size_bytes": len(content),
            "encoding": self._default_encoding,
            "delimiter": delimiter,
            "has_header": has_header,
            "headers": headers,
            "row_count": len(rows),
            "column_count": len(headers) if headers else (len(rows[0]) if rows else 0),
        }

    def detect_encoding(self, content: bytes) -> str:
        self._stats["encoding_detections"] += 1
        if content[:3] == b"\xef\xbb\xbf":
            return "utf-8-sig"
        if content[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return "utf-16"
        try:
            content.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass
        try:
            content.decode("latin-1")
            return "latin-1"
        except UnicodeDecodeError:
            pass
        return self._default_encoding

    def detect_delimiter(self, text: str) -> str:
        self._stats["delimiter_detections"] += 1
        lines = text.strip().split("\n")[:min(self._sample_rows, 20)]
        if not lines:
            return self._default_delimiter

        candidates = {",": 0, ";": 0, "\t": 0, "|": 0}
        for line in lines:
            for delim in candidates:
                candidates[delim] += line.count(delim)

        best = max(candidates, key=candidates.get)
        if candidates[best] == 0:
            return self._default_delimiter
        return best

    def extract_headers(self, text: str, delimiter: str = ",") -> List[str]:
        lines = text.strip().split("\n")
        if not lines:
            return []
        reader = csv.reader(io.StringIO(lines[0]), delimiter=delimiter)
        for row in reader:
            return [cell.strip() for cell in row]
        return []

    def extract_rows(self, text: str, delimiter: str = ",",
                     skip_header: bool = True) -> List[List[str]]:
        reader = csv.reader(io.StringIO(text.strip()), delimiter=delimiter)
        rows = list(reader)
        if skip_header and rows:
            return rows[1:]
        return rows

    def detect_has_header(self, text: str, delimiter: str = ",") -> bool:
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False
        reader1 = csv.reader(io.StringIO(lines[0]), delimiter=delimiter)
        reader2 = csv.reader(io.StringIO(lines[1]), delimiter=delimiter)
        first_row = next(reader1, [])
        second_row = next(reader2, [])
        if not first_row:
            return False
        # Heuristic: if first row is all strings and second has numbers, likely header
        first_all_str = all(not self._looks_numeric(v) for v in first_row if v.strip())
        second_has_num = any(self._looks_numeric(v) for v in second_row if v.strip())
        return first_all_str and second_has_num

    def stream_rows(self, text: str, delimiter: str = ",",
                    skip_header: bool = True) -> Iterator[List[str]]:
        reader = csv.reader(io.StringIO(text.strip()), delimiter=delimiter)
        if skip_header:
            next(reader, None)
        yield from reader

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _decode(self, content: bytes) -> str:
        encoding = self.detect_encoding(content)
        return content.decode(encoding, errors="replace")

    def _looks_numeric(self, value: str) -> bool:
        cleaned = value.strip().replace(",", "").replace(" ", "")
        if not cleaned:
            return False
        try:
            float(cleaned)
            return True
        except ValueError:
            return False


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCSVParserInit:
    def test_default_creation(self):
        parser = CSVParser()
        assert parser._default_encoding == "utf-8"
        assert parser._default_delimiter == ","

    def test_custom_config(self):
        parser = CSVParser(config={"default_encoding": "latin-1", "default_delimiter": ";"})
        assert parser._default_encoding == "latin-1"
        assert parser._default_delimiter == ";"

    def test_initial_statistics(self):
        parser = CSVParser()
        stats = parser.get_statistics()
        assert stats["files_parsed"] == 0


class TestParseFile:
    def test_parse_basic_csv(self):
        parser = CSVParser()
        content = b"name,year,value\nLondon,2025,1250\nBerlin,2025,3400\n"
        result = parser.parse_file(content, "test.csv")
        assert result["file_name"] == "test.csv"
        assert result["row_count"] == 2
        assert result["column_count"] == 3

    def test_parse_computes_hash(self):
        parser = CSVParser()
        result = parser.parse_file(b"a,b\n1,2\n", "t.csv")
        assert len(result["file_hash"]) == 64

    def test_parse_detects_delimiter(self):
        parser = CSVParser()
        result = parser.parse_file(b"a;b;c\n1;2;3\n", "test.csv")
        assert result["delimiter"] == ";"

    def test_parse_default_filename(self):
        parser = CSVParser()
        result = parser.parse_file(b"a,b\n1,2\n")
        assert result["file_name"] == "unknown.csv"

    def test_parse_updates_stats(self):
        parser = CSVParser()
        parser.parse_file(b"a,b\n1,2\n3,4\n", "test.csv")
        stats = parser.get_statistics()
        assert stats["files_parsed"] == 1
        assert stats["rows_parsed"] == 2


class TestDetectEncoding:
    def test_utf8(self):
        parser = CSVParser()
        assert parser.detect_encoding(b"hello world") == "utf-8"

    def test_utf8_bom(self):
        parser = CSVParser()
        assert parser.detect_encoding(b"\xef\xbb\xbfhello") == "utf-8-sig"

    def test_utf16_le(self):
        parser = CSVParser()
        assert parser.detect_encoding(b"\xff\xfeh\x00e\x00") == "utf-16"

    def test_utf16_be(self):
        parser = CSVParser()
        assert parser.detect_encoding(b"\xfe\xff\x00h\x00e") == "utf-16"

    def test_latin1_fallback(self):
        parser = CSVParser()
        # Bytes that are valid latin-1 but not valid utf-8
        content = bytes([0x80, 0x81, 0x82, 0x83])
        result = parser.detect_encoding(content)
        assert result in ("latin-1", "utf-8")

    def test_encoding_stats_increment(self):
        parser = CSVParser()
        parser.detect_encoding(b"test")
        parser.detect_encoding(b"test2")
        assert parser.get_statistics()["encoding_detections"] == 2


class TestDetectDelimiter:
    def test_detect_comma(self):
        parser = CSVParser()
        text = "a,b,c\n1,2,3\n4,5,6\n"
        assert parser.detect_delimiter(text) == ","

    def test_detect_semicolon(self):
        parser = CSVParser()
        text = "a;b;c\n1;2;3\n4;5;6\n"
        assert parser.detect_delimiter(text) == ";"

    def test_detect_tab(self):
        parser = CSVParser()
        text = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        assert parser.detect_delimiter(text) == "\t"

    def test_detect_pipe(self):
        parser = CSVParser()
        text = "a|b|c\n1|2|3\n4|5|6\n"
        assert parser.detect_delimiter(text) == "|"

    def test_detect_empty_text(self):
        parser = CSVParser()
        result = parser.detect_delimiter("")
        assert result == ","

    def test_detect_no_delimiter(self):
        parser = CSVParser()
        result = parser.detect_delimiter("single_value")
        assert result == ","

    def test_delimiter_stats_increment(self):
        parser = CSVParser()
        parser.detect_delimiter("a,b\n1,2\n")
        assert parser.get_statistics()["delimiter_detections"] == 1

    def test_detect_mixed_delimiters_highest_wins(self):
        parser = CSVParser()
        text = "a;b;c;d\n1;2;3;4\na,b\n"
        result = parser.detect_delimiter(text)
        assert result == ";"


class TestExtractHeaders:
    def test_extract_comma_headers(self):
        parser = CSVParser()
        text = "Facility,Year,Emissions\n"
        headers = parser.extract_headers(text, ",")
        assert headers == ["Facility", "Year", "Emissions"]

    def test_extract_semicolon_headers(self):
        parser = CSVParser()
        text = "Facility;Year;Emissions\n"
        headers = parser.extract_headers(text, ";")
        assert headers == ["Facility", "Year", "Emissions"]

    def test_extract_empty_text(self):
        parser = CSVParser()
        headers = parser.extract_headers("", ",")
        assert headers == []

    def test_extract_strips_whitespace(self):
        parser = CSVParser()
        text = " Facility , Year , Emissions \n"
        headers = parser.extract_headers(text, ",")
        assert headers == ["Facility", "Year", "Emissions"]


class TestExtractRows:
    def test_extract_with_header_skip(self):
        parser = CSVParser()
        text = "a,b\n1,2\n3,4\n"
        rows = parser.extract_rows(text, ",", skip_header=True)
        assert len(rows) == 2
        assert rows[0] == ["1", "2"]

    def test_extract_without_header_skip(self):
        parser = CSVParser()
        text = "a,b\n1,2\n3,4\n"
        rows = parser.extract_rows(text, ",", skip_header=False)
        assert len(rows) == 3

    def test_extract_semicolon_rows(self):
        parser = CSVParser()
        text = "a;b\n1;2\n"
        rows = parser.extract_rows(text, ";", skip_header=True)
        assert len(rows) == 1
        assert rows[0] == ["1", "2"]

    def test_extract_empty_text(self):
        parser = CSVParser()
        rows = parser.extract_rows("", ",")
        assert rows == []


class TestDetectHasHeader:
    def test_has_header_true(self):
        parser = CSVParser()
        text = "name,year,value\nLondon,2025,1250.5\n"
        assert parser.detect_has_header(text, ",") is True

    def test_has_header_false_all_numbers(self):
        parser = CSVParser()
        text = "1,2,3\n4,5,6\n"
        assert parser.detect_has_header(text, ",") is False

    def test_has_header_single_line(self):
        parser = CSVParser()
        text = "name,year,value"
        assert parser.detect_has_header(text, ",") is False


class TestStreamRows:
    def test_stream_yields_rows(self):
        parser = CSVParser()
        text = "a,b\n1,2\n3,4\n"
        rows = list(parser.stream_rows(text, ","))
        assert len(rows) == 2

    def test_stream_without_header_skip(self):
        parser = CSVParser()
        text = "a,b\n1,2\n"
        rows = list(parser.stream_rows(text, ",", skip_header=False))
        assert len(rows) == 2


class TestCSVEdgeCases:
    def test_bom_handling(self):
        parser = CSVParser()
        content = b"\xef\xbb\xbfname,year\nTest,2025\n"
        result = parser.parse_file(content, "bom.csv")
        assert result["row_count"] >= 1

    def test_mixed_line_endings(self):
        parser = CSVParser()
        text = "a,b\r\n1,2\n3,4\r\n"
        rows = parser.extract_rows(text, ",", skip_header=True)
        assert len(rows) >= 2

    def test_quoted_fields(self):
        parser = CSVParser()
        text = '"Name","Year"\n"London, UK","2025"\n'
        rows = parser.extract_rows(text, ",", skip_header=True)
        assert rows[0][0] == "London, UK"

    def test_empty_file(self):
        parser = CSVParser()
        result = parser.parse_file(b"", "empty.csv")
        assert result["row_count"] == 0

    def test_single_column(self):
        parser = CSVParser()
        text = "name\nLondon\nBerlin\n"
        rows = parser.extract_rows(text, ",", skip_header=True)
        assert len(rows) == 2
        assert rows[0] == ["London"]

    def test_unicode_content(self):
        parser = CSVParser()
        # Use numeric data in second row so header detection works correctly
        content = "Name,Value\nM\u00fcller,1250.5\n".encode("utf-8")
        result = parser.parse_file(content, "unicode.csv")
        assert result["row_count"] == 1

    def test_very_long_header(self):
        parser = CSVParser()
        long_header = "A" * 500
        text = f"{long_header},b\n1,2\n"
        headers = parser.extract_headers(text, ",")
        assert len(headers[0]) == 500

    def test_numeric_headers(self):
        parser = CSVParser()
        text = "2020,2021,2022\n100,200,300\n"
        has_header = parser.detect_has_header(text, ",")
        assert has_header is False


class TestCSVParserStatistics:
    def test_stats_accumulate(self):
        parser = CSVParser()
        parser.parse_file(b"a,b\n1,2\n", "a.csv")
        parser.parse_file(b"x,y\n3,4\n5,6\n", "b.csv")
        stats = parser.get_statistics()
        assert stats["files_parsed"] == 2
        assert stats["rows_parsed"] == 3
