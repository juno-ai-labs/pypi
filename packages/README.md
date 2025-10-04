# Packages Directory

This directory contains container-specific subdirectories, each with Python packages built for a specific Docker container image.

## Structure

Each subdirectory represents a Docker container build environment:
- `packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/` - Packages for NVIDIA L4T JetPack r36.4.0
- Each directory contains an `info.yaml` with container metadata

## Supported formats
- `.whl` - Python wheel files
- `.tar.gz` - Source distributions
- `.zip` - Zip source distributions

## How to add packages

1. Add your package files to the appropriate container directory
2. Commit and push to the `main` branch
3. The CI/CD workflow will automatically generate separate PyPI indexes for each container and deploy them to GitHub Pages

## Example

```bash
# Add a wheel file to a specific container directory
cp dist/mypackage-1.0.0-cp310-cp310-linux_aarch64.whl packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/

# Commit and push
git add packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/mypackage-1.0.0-cp310-cp310-linux_aarch64.whl
git commit -m "Add mypackage 1.0.0 for nvcr-io-nvidia-l4t-jetpack-r36-4-0"
git push
```

## Installing packages

Use the container-specific index URL when installing:

```bash
pip install --index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ mypackage
```
