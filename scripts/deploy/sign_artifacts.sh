#!/bin/bash
# GreenLang CLI Artifact Signing Script
# This script signs all distribution artifacts using cosign keyless signing

set -e

echo "🔏 GreenLang CLI Artifact Signing Script"
echo "=========================================="

# Check if cosign is installed
if ! command -v cosign &> /dev/null; then
    echo "❌ Error: cosign is not installed. Please install cosign first."
    echo "   Install: go install github.com/sigstore/cosign/v2/cmd/cosign@latest"
    exit 1
fi

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist/ directory not found. Please build the package first."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")"

echo "📁 Checking dist/ directory contents..."
ls -la dist/

# Find all .whl and .tar.gz files
WHEEL_FILES=$(find dist/ -name "*.whl" -type f)
TARBALL_FILES=$(find dist/ -name "*.tar.gz" -type f)

if [ -z "$WHEEL_FILES" ] && [ -z "$TARBALL_FILES" ]; then
    echo "❌ Error: No .whl or .tar.gz files found in dist/"
    exit 1
fi

echo "🔍 Found artifacts to sign:"
for file in $WHEEL_FILES $TARBALL_FILES; do
    echo "  - $file"
done

echo ""
echo "🔐 Starting keyless signing with cosign..."
echo "   Note: This will require OIDC authentication (GitHub, Google, etc.)"
echo ""

# Sign each artifact
for file in $WHEEL_FILES $TARBALL_FILES; do
    echo "📝 Signing: $file"

    # Sign the file with cosign keyless signing
    cosign sign-blob --yes "$file" --output-signature "${file}.sig"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully signed: $file"
        echo "   Signature created: ${file}.sig"
    else
        echo "❌ Failed to sign: $file"
        exit 1
    fi

    echo ""
done

echo "🎯 Creating attestations..."
for file in $WHEEL_FILES $TARBALL_FILES; do
    echo "📋 Creating attestation for: $file"

    # Create SLSA provenance attestation
    cosign attest --yes --predicate provenance.txt --type slsaprovenance "$file"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully created attestation for: $file"
    else
        echo "⚠️  Warning: Failed to create attestation for: $file"
        # Don't exit on attestation failure, continue with other files
    fi

    echo ""
done

echo "📊 Summary of signed artifacts:"
echo "================================"
for file in $WHEEL_FILES $TARBALL_FILES; do
    if [ -f "${file}.sig" ]; then
        echo "✅ $file"
        echo "   └─ Signature: ${file}.sig"

        # Verify the signature
        if cosign verify-blob --signature "${file}.sig" "$file" &>/dev/null; then
            echo "   └─ ✅ Signature verified"
        else
            echo "   └─ ⚠️  Signature verification failed"
        fi
    else
        echo "❌ $file (signature missing)"
    fi
done

echo ""
echo "🎉 Artifact signing completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Verify all signatures are present"
echo "   2. Test signature verification"
echo "   3. Include signatures in release"
echo "   4. Update documentation"
echo ""
echo "🔍 To verify a signature manually:"
echo "   cosign verify-blob --signature <file>.sig <file>"
echo ""
echo "📚 For more information:"
echo "   - Cosign documentation: https://docs.sigstore.dev/"
echo "   - SLSA provenance: https://slsa.dev/"