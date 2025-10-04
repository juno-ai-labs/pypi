# Setup Instructions

This guide will help you set up your private PyPI repository using GitHub Pages.

## Prerequisites

- A GitHub account
- Admin access to this repository
- Python packages (wheels or source distributions) to publish

## Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (in the repository menu)
3. In the left sidebar, click on **Pages**
4. Under **Source**, select **GitHub Actions** from the dropdown
5. Click **Save**

That's it! GitHub Pages is now configured to deploy from GitHub Actions.

## Step 2: Verify the Workflow

The GitHub Actions workflow is already set up in `.github/workflows/build-index.yml`.

To verify it's working:

1. Go to the **Actions** tab in your repository
2. You should see a workflow called "Build PyPI Index"
3. If there are any workflow runs, check their status

## Step 3: Add Your First Package

### Option A: Upload an existing package

If you have a package already built:

```bash
# Clone the repository
git clone https://github.com/juno-ai-labs/pypi.git
cd pypi

# Copy your package
cp /path/to/your/package.whl packages/

# Commit and push
git add packages/your-package.whl
git commit -m "Add your-package"
git push origin main
```

### Option B: Build and upload a new package

If you're developing a new package:

```bash
# In your package directory
python setup.py sdist bdist_wheel

# Clone this repository
git clone https://github.com/juno-ai-labs/pypi.git
cd pypi

# Copy the built package
cp /path/to/your/project/dist/*.whl packages/

# Commit and push
git add packages/*.whl
git commit -m "Add your-package"
git push origin main
```

## Step 4: Wait for Deployment

After pushing, the workflow will automatically:

1. **Trigger**: The workflow detects changes in the `packages/` directory
2. **Build**: It generates the PyPI index using `dumb-pypi`
3. **Deploy**: It publishes the index to GitHub Pages

This usually takes 1-2 minutes. You can monitor progress in the **Actions** tab.

## Step 5: Find Your PyPI URL

Once deployed, your PyPI index will be available at:

```
https://<username>.github.io/<repository-name>/
```

For this repository:
```
https://juno-ai-labs.github.io/pypi/
```

## Step 6: Install Packages

Test your private PyPI by installing a package:

```bash
pip install --index-url https://juno-ai-labs.github.io/pypi/ your-package
```

## Troubleshooting

### "404 Not Found" when accessing the GitHub Pages URL

**Problem**: The GitHub Pages site is not yet published or is disabled.

**Solution**:
1. Check that GitHub Pages is enabled (Step 1)
2. Wait a few minutes after the first deployment
3. Check the Actions tab for any failed workflows

### Workflow not triggering

**Problem**: The workflow doesn't run when you push packages.

**Solution**:
1. Ensure you're pushing to the `main` branch
2. Verify your changes include files in the `packages/` directory
3. Manually trigger the workflow:
   - Go to Actions → "Build PyPI Index" → "Run workflow"

### Package not appearing in the index

**Problem**: You uploaded a package but it doesn't show in the index.

**Solution**:
1. Verify the file extension is `.whl`, `.tar.gz`, or `.zip`
2. Check that the file is directly in the `packages/` directory (not in a subdirectory)
3. Look at the workflow logs for any errors

### "Permission denied" when pushing to repository

**Problem**: You can't push changes to the repository.

**Solution**:
1. Verify you have write access to the repository
2. Check that you've cloned with the correct URL
3. Ensure your Git credentials are configured correctly

## Advanced Configuration

### Custom Package URL

If you want to host packages elsewhere (e.g., S3), you can modify the workflow:

Edit `.github/workflows/build-index.yml` and change the `--packages-url` parameter:

```yaml
- name: Build PyPI index
  run: |
    dumb-pypi \
      --package-list package-list.txt \
      --packages-url https://your-cdn.example.com/packages/ \
      --output-dir index
```

### Custom Index Title

Add `--title` to the dumb-pypi command:

```yaml
- name: Build PyPI index
  run: |
    dumb-pypi \
      --package-list package-list.txt \
      --packages-url https://raw.githubusercontent.com/${{ github.repository }}/main/packages/ \
      --output-dir index \
      --title "My Private PyPI"
```

### Adding a Logo

Add `--logo` and `--logo-width` to the dumb-pypi command:

```yaml
- name: Build PyPI index
  run: |
    dumb-pypi \
      --package-list package-list.txt \
      --packages-url https://raw.githubusercontent.com/${{ github.repository }}/main/packages/ \
      --output-dir index \
      --logo "https://example.com/logo.png" \
      --logo-width 200
```

## Security Considerations

### Private Repository

If your repository is private:
- The PyPI index on GitHub Pages will be **public** by default
- The package files in the `packages/` directory will be **private**
- Users won't be able to download packages without repository access

To make packages accessible:
1. Host packages on a CDN or public storage
2. Update the `--packages-url` parameter accordingly

### Public Repository

If your repository is public:
- Both the index and packages will be publicly accessible
- Anyone can install your packages
- Consider using GitHub Releases for better package hosting

## Next Steps

- Read [USAGE.md](USAGE.md) for detailed usage instructions
- Read [README.md](README.md) for an overview of the repository
- Check the [dumb-pypi documentation](https://github.com/chriskuehl/dumb-pypi) for more options
