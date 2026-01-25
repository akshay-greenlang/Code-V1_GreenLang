#!/usr/bin/env python3
"""
Proto Stub Generator for GL-003 UNIFIEDSTEAM

Generates Python gRPC stubs from proto definitions.
Requires: grpcio-tools

Usage:
    python generate_stubs.py
    python generate_stubs.py --output-dir ../grpc_generated

Author: GL-003 DevOps Team
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def get_proto_files(proto_dir: Path) -> List[Path]:
    """Get all proto files in directory."""
    return list(proto_dir.glob("*.proto"))


def generate_python_stubs(
    proto_files: List[Path],
    proto_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Generate Python stubs from proto files.

    Args:
        proto_files: List of proto file paths
        proto_dir: Directory containing proto files
        output_dir: Output directory for generated files

    Returns:
        True if successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    init_file = output_dir / "__init__.py"
    init_file.write_text('"""Generated gRPC stubs for GL-003 UNIFIEDSTEAM."""\n')

    for proto_file in proto_files:
        print(f"Generating stubs for: {proto_file.name}")

        # Generate Python protobuf files
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            f"--pyi_out={output_dir}",  # Type stubs
            str(proto_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"Error generating stubs for {proto_file.name}:")
                print(result.stderr)
                return False

        except FileNotFoundError:
            print("Error: grpcio-tools not installed.")
            print("Install with: pip install grpcio-tools")
            return False

    return True


def verify_stubs(output_dir: Path) -> bool:
    """Verify generated stubs are importable."""
    pb2_files = list(output_dir.glob("*_pb2.py"))
    grpc_files = list(output_dir.glob("*_pb2_grpc.py"))

    print(f"\nGenerated files:")
    print(f"  - {len(pb2_files)} _pb2.py files (protobuf messages)")
    print(f"  - {len(grpc_files)} _pb2_grpc.py files (gRPC services)")

    # Try to import
    sys.path.insert(0, str(output_dir.parent))

    for pb2_file in pb2_files:
        module_name = pb2_file.stem
        try:
            __import__(f"{output_dir.name}.{module_name}")
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
            return False

    return True


def create_stub_wrapper(output_dir: Path, proto_name: str) -> None:
    """Create a wrapper module for easier imports."""
    wrapper_content = f'''"""
Wrapper module for {proto_name} gRPC stubs.

Provides convenient imports for generated protobuf and gRPC classes.
"""

from .{proto_name}_pb2 import *
from .{proto_name}_pb2_grpc import *

# Service stubs for client use
__all__ = [
    # Services
    "SteamPropertiesServiceStub",
    "OptimizationServiceStub",
    "DiagnosticsServiceStub",
    "StreamingServiceStub",
    "RCAServiceStub",

    # Common messages
    "SteamState",
    "Recommendation",
    "CausalFactor",
    "Uncertainty",

    # Request/Response messages
    "SteamPropertiesRequest",
    "SteamPropertiesResponse",
    "OptimizationRequest",
    "OptimizationResponse",
    "TrapDiagnosticsRequest",
    "TrapDiagnosticsResponse",
    "RCARequest",
    "RCAResponse",
]
'''

    wrapper_file = output_dir / f"{proto_name}_wrapper.py"
    wrapper_file.write_text(wrapper_content)
    print(f"Created wrapper: {wrapper_file.name}")


def print_service_summary(proto_file: Path) -> None:
    """Print summary of services defined in proto file."""
    content = proto_file.read_text()

    services = []
    rpcs = []

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("service "):
            service_name = line.split()[1].rstrip("{")
            services.append(service_name)
        elif line.startswith("rpc "):
            rpc_name = line.split("(")[0].replace("rpc ", "").strip()
            rpcs.append(rpc_name)

    print(f"\nProto file summary: {proto_file.name}")
    print(f"  Services: {len(services)}")
    for svc in services:
        print(f"    - {svc}")
    print(f"  RPC methods: {len(rpcs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python gRPC stubs from proto files"
    )
    parser.add_argument(
        "--proto-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing proto files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "grpc_generated",
        help="Output directory for generated stubs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated stubs are importable",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print proto file summary",
    )

    args = parser.parse_args()

    proto_files = get_proto_files(args.proto_dir)

    if not proto_files:
        print(f"No proto files found in: {args.proto_dir}")
        sys.exit(1)

    print(f"Found {len(proto_files)} proto file(s)")

    if args.summary:
        for proto_file in proto_files:
            print_service_summary(proto_file)
        return

    # Generate stubs
    success = generate_python_stubs(
        proto_files,
        args.proto_dir,
        args.output_dir,
    )

    if not success:
        print("\nStub generation failed!")
        sys.exit(1)

    # Create wrapper modules
    for proto_file in proto_files:
        proto_name = proto_file.stem
        create_stub_wrapper(args.output_dir, proto_name)

    print(f"\nStubs generated successfully in: {args.output_dir}")

    # Verify if requested
    if args.verify:
        print("\nVerifying generated stubs...")
        if verify_stubs(args.output_dir):
            print("All stubs verified successfully!")
        else:
            print("Stub verification failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
