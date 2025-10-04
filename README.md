# pypi

A private PyPI repository hosted on GitHub Pages using [dumb-pypi](https://github.com/chriskuehl/dumb-pypi).

## Quick Start

1. **Enable GitHub Pages**: See [SETUP.md](SETUP.md) for detailed setup instructions
2. **Add packages**: Copy your `.whl` or `.tar.gz` files to the appropriate container directory under `packages/` (e.g., `packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/`)
3. **Push to main**: The CI/CD workflow will automatically build and deploy the indexes
4. **Install packages**: Use `pip install --index-url https://juno-ai-labs.github.io/pypi/<container-name>/ your-package`

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup guide for first-time configuration
- **[USAGE.md](USAGE.md)** - Detailed usage instructions and examples
- **[packages/README.md](packages/README.md)** - Quick reference for adding packages

## Overview

This repository automatically generates and publishes PyPI-compatible package indexes for different Docker container builds. Each container directory under `packages/` gets its own PyPI index, allowing you to install packages built specifically for your container environment.

## How It Works

1. **Upload packages**: Add your `.whl`, `.tar.gz`, or `.zip` files to a container-specific directory under `packages/` (e.g., `packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/`)
2. **Automatic indexing**: GitHub Actions automatically generates separate PyPI indexes for each container using `dumb-pypi`
3. **Publishing**: The indexes are deployed to GitHub Pages with a root navigation page
4. **Installation**: Use pip to install packages from the appropriate container-specific index

## Adding Packages

```bash
# Copy your package to the appropriate container directory
cp dist/mypackage-1.0.0-py3-none-any.whl packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/

# Commit and push
git add packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/mypackage-1.0.0-py3-none-any.whl
git commit -m "Add mypackage 1.0.0 for nvcr-io-nvidia-l4t-jetpack-r36-4-0"
git push
```

The CI/CD workflow will automatically:
- Scan each container directory under `packages/` for package files
- Generate separate indexes for each container
- Create a root navigation page to browse available containers
- Deploy to GitHub Pages

## Installing Packages

Once the indexes are published, you can install packages using pip with the container-specific index:

```bash
# Install a specific package from a container-specific index
pip install --index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ mypackage

# Use as an extra index alongside PyPI
pip install --extra-index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ mypackage
```

Or add to your `requirements.txt`:

```
--extra-index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/
mypackage==1.0.0
```

Or configure in `pip.conf` or `~/.pip/pip.conf`:

```ini
[global]
extra-index-url = https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/
```

## Configuration

The PyPI indexes are built using the following configuration:
- **Package lists**: Automatically generated from files in each container directory
- **Package URLs**: Raw GitHub content URLs for package files in each container directory
- **Output structure**: `index/` with subdirectories for each container (deployed to GitHub Pages)
- **Container metadata**: Read from `info.yaml` in each container directory

## GitHub Pages Setup

To enable GitHub Pages for this repository:

1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"
3. The workflow will automatically deploy the index

## Workflow

The CI/CD workflow (`.github/workflows/build-index.yml`) triggers on:
- Pushes to the `main` branch that modify files in `packages/`
- Manual workflow dispatch

## Supported Package Formats

- `.whl` - Python wheel files
- `.tar.gz` - Gzipped tar source distributions
- `.zip` - Zip source distributions

## Example

See the example in the problem statement:

```bash
$ dumb-pypi \
    --package-list pkg.txt \
    --packages-url https://my-pypi-packages.s3.amazonaws.com/ \
    --output-dir my-built-index
```

This repository automates this process using GitHub Actions and GitHub Pages.