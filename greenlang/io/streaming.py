"""
GreenLang Streaming I/O
Memory-efficient streaming for large datasets.
"""

from typing import Iterator, Any, Dict, Optional, Union, Callable
from pathlib import Path
import logging
import json
import csv

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Memory-efficient streaming reader for large files.

    Supports:
    - Line-by-line text reading
    - JSON streaming (JSONL format)
    - CSV streaming
    - Chunked reading
    - Progress callbacks

    Example:
        reader = StreamReader()

        # Stream JSON lines
        for record in reader.stream_jsonl("large_file.jsonl"):
            process(record)

        # Stream CSV with chunks
        for chunk in reader.stream_csv_chunks("data.csv", chunk_size=1000):
            batch_process(chunk)
    """

    def __init__(self, encoding: str = "utf-8", buffer_size: int = 8192):
        """
        Initialize stream reader.

        Args:
            encoding: Text encoding
            buffer_size: Buffer size for reading
        """
        self.encoding = encoding
        self.buffer_size = buffer_size

    def stream_lines(
        self,
        file_path: Union[str, Path],
        skip_empty: bool = True
    ) -> Iterator[str]:
        """
        Stream file line by line.

        Args:
            file_path: Path to file
            skip_empty: Skip empty lines

        Yields:
            Lines from file
        """
        path = Path(file_path)

        with open(path, 'r', encoding=self.encoding, buffering=self.buffer_size) as f:
            for line in f:
                line = line.rstrip('\n\r')
                if skip_empty and not line:
                    continue
                yield line

    def stream_jsonl(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        Stream JSONL (JSON Lines) file.

        Args:
            file_path: Path to JSONL file

        Yields:
            Parsed JSON objects
        """
        for line in self.stream_lines(file_path):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {str(e)}")
                continue

    def stream_csv(
        self,
        file_path: Union[str, Path],
        delimiter: str = ",",
        has_header: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream CSV file record by record.

        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            has_header: Whether CSV has header row

        Yields:
            Records as dictionaries
        """
        path = Path(file_path)

        with open(path, 'r', encoding=self.encoding, newline='') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
            else:
                reader = csv.reader(f, delimiter=delimiter)

            for row in reader:
                if has_header:
                    yield dict(row)
                else:
                    yield list(row)

    def stream_csv_chunks(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        delimiter: str = ",",
        has_header: bool = True
    ) -> Iterator[list]:
        """
        Stream CSV file in chunks.

        Args:
            file_path: Path to CSV file
            chunk_size: Number of records per chunk
            delimiter: CSV delimiter
            has_header: Whether CSV has header row

        Yields:
            Lists of records (chunks)
        """
        chunk = []

        for record in self.stream_csv(file_path, delimiter, has_header):
            chunk.append(record)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        # Yield remaining records
        if chunk:
            yield chunk

    def stream_with_progress(
        self,
        file_path: Union[str, Path],
        callback: Callable[[int], None],
        callback_interval: int = 1000
    ) -> Iterator[str]:
        """
        Stream lines with progress callback.

        Args:
            file_path: Path to file
            callback: Function called with line count
            callback_interval: Lines between callbacks

        Yields:
            Lines from file
        """
        count = 0

        for line in self.stream_lines(file_path):
            count += 1

            if count % callback_interval == 0:
                callback(count)

            yield line

        # Final callback
        callback(count)


class StreamWriter:
    """
    Memory-efficient streaming writer.

    Supports:
    - Line-by-line writing
    - JSON streaming (JSONL format)
    - CSV streaming
    - Append mode

    Example:
        writer = StreamWriter()

        # Write JSONL
        with writer.open_jsonl("output.jsonl") as f:
            for record in large_dataset:
                f.write(record)

        # Stream CSV
        with writer.open_csv("output.csv", fieldnames=["id", "name"]) as f:
            for record in records:
                f.write(record)
    """

    def __init__(self, encoding: str = "utf-8", buffer_size: int = 8192):
        """
        Initialize stream writer.

        Args:
            encoding: Text encoding
            buffer_size: Buffer size for writing
        """
        self.encoding = encoding
        self.buffer_size = buffer_size

    def open_jsonl(self, file_path: Union[str, Path], mode: str = "w"):
        """
        Open JSONL file for streaming writes.

        Args:
            file_path: Path to output file
            mode: File mode ('w' or 'a')

        Returns:
            JSONLWriter context manager
        """
        return JSONLWriter(file_path, mode, self.encoding)

    def open_csv(
        self,
        file_path: Union[str, Path],
        fieldnames: list,
        mode: str = "w",
        delimiter: str = ","
    ):
        """
        Open CSV file for streaming writes.

        Args:
            file_path: Path to output file
            fieldnames: CSV column names
            mode: File mode ('w' or 'a')
            delimiter: CSV delimiter

        Returns:
            CSVWriter context manager
        """
        return CSVWriter(file_path, fieldnames, mode, delimiter, self.encoding)


class JSONLWriter:
    """Context manager for streaming JSONL writes."""

    def __init__(self, file_path: Union[str, Path], mode: str, encoding: str):
        self.file_path = Path(file_path)
        self.mode = mode
        self.encoding = encoding
        self.file = None
        self.count = 0

    def __enter__(self):
        # Create parent directory if needed
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.file_path, self.mode, encoding=self.encoding)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write(self, record: Dict[str, Any]):
        """Write a record as JSON line."""
        json_line = json.dumps(record)
        self.file.write(json_line + '\n')
        self.count += 1

    def get_count(self) -> int:
        """Get number of records written."""
        return self.count


class CSVWriter:
    """Context manager for streaming CSV writes."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fieldnames: list,
        mode: str,
        delimiter: str,
        encoding: str
    ):
        self.file_path = Path(file_path)
        self.fieldnames = fieldnames
        self.mode = mode
        self.delimiter = delimiter
        self.encoding = encoding
        self.file = None
        self.writer = None
        self.count = 0

    def __enter__(self):
        # Create parent directory if needed
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.file_path, self.mode, encoding=self.encoding, newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, delimiter=self.delimiter)

        # Write header if new file
        if self.mode == 'w':
            self.writer.writeheader()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write(self, record: Dict[str, Any]):
        """Write a record to CSV."""
        self.writer.writerow(record)
        self.count += 1

    def get_count(self) -> int:
        """Get number of records written."""
        return self.count
