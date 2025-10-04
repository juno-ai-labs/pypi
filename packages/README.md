# Packages Directory

Place your Python packages (wheels and source distributions) in this directory.

## Supported formats
- `.whl` - Python wheel files
- `.tar.gz` - Source distributions
- `.zip` - Zip source distributions

## How to add packages

1. Add your package files to this directory
2. Commit and push to the `main` branch
3. The CI/CD workflow will automatically generate the PyPI index and deploy it to GitHub Pages

## Example

```bash
# Add a wheel file
cp dist/mypackage-1.0.0-py3-none-any.whl packages/

# Commit and push
git add packages/mypackage-1.0.0-py3-none-any.whl
git commit -m "Add mypackage 1.0.0"
git push
```
