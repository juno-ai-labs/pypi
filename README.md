# pypi

A private PyPI repository hosted on GitHub Pages using [dumb-pypi](https://github.com/chriskuehl/dumb-pypi).

## Quick Start

1. **Enable GitHub Pages**: See [SETUP.md](SETUP.md) for detailed setup instructions
2. **Add packages**: Copy your `.whl` or `.tar.gz` files to the `packages/` directory
3. **Push to main**: The CI/CD workflow will automatically build and deploy the index
4. **Install packages**: Use `pip install --index-url https://juno-ai-labs.github.io/pypi/ your-package`

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup guide for first-time configuration
- **[USAGE.md](USAGE.md)** - Detailed usage instructions and examples
- **[packages/README.md](packages/README.md)** - Quick reference for adding packages

## Overview

This repository automatically generates and publishes a PyPI-compatible package index whenever you upload Python packages (wheels or source distributions) to the `packages/` directory.

## How It Works

1. **Upload packages**: Add your `.whl`, `.tar.gz`, or `.zip` files to the `packages/` directory
2. **Automatic indexing**: GitHub Actions automatically generates a PyPI index using `dumb-pypi`
3. **Publishing**: The index is deployed to GitHub Pages
4. **Installation**: Use pip to install packages from your private index

## Adding Packages

```bash
# Copy your package to the packages directory
cp dist/mypackage-1.0.0-py3-none-any.whl packages/

# Commit and push
git add packages/mypackage-1.0.0-py3-none-any.whl
git commit -m "Add mypackage 1.0.0"
git push
```

The CI/CD workflow will automatically:
- Scan the `packages/` directory for all package files
- Generate a package list
- Build the PyPI index
- Deploy to GitHub Pages

## Installing Packages

Once the index is published, you can install packages using pip:

```bash
# Install a specific package
pip install --index-url https://juno-ai-labs.github.io/pypi/ mypackage

# Use as an extra index alongside PyPI
pip install --extra-index-url https://juno-ai-labs.github.io/pypi/ mypackage
```

Or add to your `requirements.txt`:

```
--extra-index-url https://juno-ai-labs.github.io/pypi/
mypackage==1.0.0
```

Or configure in `pip.conf` or `~/.pip/pip.conf`:

```ini
[global]
extra-index-url = https://juno-ai-labs.github.io/pypi/
```

## Configuration

The PyPI index is built using the following configuration:
- **Package list**: Automatically generated from files in `packages/`
- **Package URL**: Raw GitHub content URL for package files
- **Output directory**: `index/` (deployed to GitHub Pages)

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