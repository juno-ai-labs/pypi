# Usage Guide

## Adding Your First Package

### 1. Build your package

```bash
# In your Python project directory
python setup.py sdist bdist_wheel
# or with poetry
poetry build
# or with flit
flit build
```

### 2. Copy the package to this repository

```bash
# Clone this repository
git clone https://github.com/juno-ai-labs/pypi.git
cd pypi

# Copy your wheel and/or source distribution to the appropriate container directory
cp /path/to/your/project/dist/yourpackage-1.0.0-py3-none-any.whl packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/

# Commit and push
git add packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/yourpackage-1.0.0-py3-none-any.whl
git commit -m "Add yourpackage 1.0.0 for nvcr-io-nvidia-l4t-jetpack-r36-4-0"
git push origin main
```

### 3. Wait for GitHub Actions

The workflow will automatically:
- Detect the new package in the container directory
- Generate separate PyPI indexes for each container
- Deploy to GitHub Pages (usually takes 1-2 minutes)

### 4. Install your package

Once deployed, you can install your package using the container-specific index:

```bash
pip install --index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ yourpackage
```

## Using with pip

### Option 1: Command line

```bash
# Install from your private index only
pip install --index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ yourpackage

# Use your private index as a fallback to PyPI
pip install --extra-index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ yourpackage
```

### Option 2: requirements.txt

Add this at the top of your `requirements.txt`:

```
--extra-index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/
yourpackage==1.0.0
requests==2.31.0
```

### Option 3: pip.conf

Create or edit `~/.pip/pip.conf` (Linux/macOS) or `%APPDATA%\pip\pip.ini` (Windows):

```ini
[global]
extra-index-url = https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/
```

### Option 4: Environment variable

```bash
export PIP_EXTRA_INDEX_URL=https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/
pip install yourpackage
```

## Using with Poetry

Add to your `pyproject.toml`:

```toml
[[tool.poetry.source]]
name = "private-pypi"
url = "https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/"
priority = "supplemental"
```

Then install:

```bash
poetry add yourpackage
```

## Using with Pipenv

```bash
pipenv install --extra-index-url https://juno-ai-labs.github.io/pypi/nvcr-io-nvidia-l4t-jetpack-r36-4-0/ yourpackage
```

## Updating Packages

To update a package, simply add the new version to the appropriate container directory:

```bash
# Add new version
cp dist/yourpackage-1.1.0-py3-none-any.whl packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/

# Commit and push
git add packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/yourpackage-1.1.0-py3-none-any.whl
git commit -m "Update yourpackage to 1.1.0"
git push origin main
```

Both versions will be available in the index. Pip will automatically select the latest version unless you specify otherwise.

## Removing Packages

To remove a package from the index:

```bash
# Remove the package file
git rm packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/yourpackage-1.0.0-py3-none-any.whl

# Commit and push
git commit -m "Remove yourpackage 1.0.0"
git push origin main
```

The index will be regenerated without that package.

## Troubleshooting

### GitHub Pages not enabled

1. Go to repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. Save the settings

### Workflow not triggering

The workflow triggers on:
- Pushes to `main` branch that modify files in `packages/**`
- Manual trigger via Actions tab → "Build PyPI Index" → "Run workflow"

### Package not appearing in index

1. Check that the file is in a container directory under `packages/` (e.g., `packages/nvcr-io-nvidia-l4t-jetpack-r36-4-0/`)
2. Verify the file has a supported extension (`.whl`, `.tar.gz`, `.zip`)
3. Check the GitHub Actions workflow for errors
4. Wait a few minutes for GitHub Pages to update

### Cannot install package

1. Verify the GitHub Pages site is published and accessible
2. Check that you're using the correct container-specific URL: `https://juno-ai-labs.github.io/pypi/<container-name>/`
3. Try with `--index-url` first to test, then switch to `--extra-index-url`
