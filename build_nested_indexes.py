#!/usr/bin/env python3
"""
Build nested PyPI indexes for each container directory in packages/.
Each directory represents a docker container with its own set of packages.
"""
import os
import sys
import subprocess
from pathlib import Path
import yaml


def get_container_dirs(packages_dir):
    """Get all container directories in packages/ that contain packages."""
    container_dirs = []
    packages_path = Path(packages_dir)
    
    for item in packages_path.iterdir():
        if item.is_dir() and item.name != '__pycache__':
            # Check if directory contains any package files
            has_packages = any(
                item.glob('*.whl') or 
                item.glob('*.tar.gz') or 
                item.glob('*.zip')
            )
            if has_packages:
                container_dirs.append(item.name)
    
    return sorted(container_dirs)


def build_index_for_container(container_name, base_url, output_dir):
    """Build PyPI index for a specific container directory."""
    packages_dir = Path('packages') / container_name
    output_path = Path(output_dir) / container_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create package list for this container
    package_list_file = output_path / 'package-list.txt'
    packages = []
    
    for ext in ['*.whl', '*.tar.gz', '*.zip']:
        packages.extend([f.name for f in packages_dir.glob(ext)])
    
    packages.sort()
    
    with open(package_list_file, 'w') as f:
        f.write('\n'.join(packages))
    
    print(f"Building index for {container_name} with {len(packages)} packages")
    
    # Load info.yaml if it exists
    info_file = packages_dir / 'info.yaml'
    title = container_name
    if info_file.exists():
        with open(info_file) as f:
            info = yaml.safe_load(f)
            if info and 'base_image' in info:
                title = f"{container_name} ({info['base_image']})"
    
    # Build the index using dumb-pypi
    packages_url = f"{base_url}/packages/{container_name}/"
    
    cmd = [
        'dumb-pypi',
        '--package-list', str(package_list_file),
        '--packages-url', packages_url,
        '--output-dir', str(output_path),
        '--title', title
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building index for {container_name}:")
        print(result.stderr)
        sys.exit(1)
    
    # Clean up package list file
    package_list_file.unlink()
    
    return {
        'name': container_name,
        'title': title,
        'package_count': len(packages)
    }


def create_root_index(containers_info, output_dir):
    """Create root index.html to navigate nested indexes."""
    html_content = """<!doctype html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Juno Labs PyPI</title>
    <style>
        /* CSS Reset */
        html, body, div, span, applet, object, iframe,
        h1, h2, h3, h4, h5, h6, p, blockquote, pre,
        a, abbr, acronym, address, big, cite, code,
        del, dfn, em, img, ins, kbd, q, s, samp,
        small, strike, strong, sub, sup, tt, var,
        b, u, i, center,
        dl, dt, dd, ol, ul, li,
        fieldset, form, label, legend,
        table, caption, tbody, tfoot, thead, tr, th, td,
        article, aside, canvas, details, embed,
        figure, figcaption, footer, header, hgroup,
        menu, nav, output, ruby, section, summary,
        time, mark, audio, video {
            margin: 0;
            padding: 0;
            border: 0;
            font-size: 100%;
            font: inherit;
            vertical-align: baseline;
        }
        article, aside, details, figcaption, figure,
        footer, header, hgroup, menu, nav, section {
            display: block;
        }
        body {
            line-height: 1;
        }
        ol, ul {
            list-style: none;
        }
        blockquote, q {
            quotes: none;
        }
        blockquote:before, blockquote:after,
        q:before, q:after {
            content: '';
            content: none;
        }
        table {
            border-collapse: collapse;
            border-spacing: 0;
        }

        /* Main Styles - matching dumb-pypi */
        body, html {
            padding: 0;
            margin: 0;
            font-family: Helvetica, Arial, sans-serif;
        }

        .width {
            max-width: 960px;
            margin: 0 auto;
        }

        .container {
            margin-top: 75px;
        }

        .header-container {
            background-color: #e6f8ff;
            padding: 20px;
            border-bottom: solid 1px #e2e2e2;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }

        .header {
            display: table;
            width: 100%;
        }

        .title {
            display: table-cell;
            vertical-align: middle;
        }

        .title h1 {
            margin: 0 !important;
            font-weight: bold;
            font-size: 24px;
        }

        p, h2, h3 {
            margin-bottom: 10px;
        }

        h2 {
            font-weight: bold;
            font-size: 20px;
            margin-top: 20px;
        }

        strong {
            font-weight: bold;
        }

        code {
            font-family: 'Courier New', monospace;
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .package-list {
            margin-top: 20px;
        }

        .package {
            display: block;
            font-size: 14px;
            padding: 10px;
            color: #111;
            text-decoration: none;
            border-bottom: 1px solid #f0f0f0;
        }

        .package:hover {
            background-color: #ebf1ff;
        }

        .package strong {
            color: #0066cc;
        }

        .package-meta {
            color: #666;
            font-size: 13px;
            margin-left: 10px;
        }

        .intro {
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #0066cc;
            margin-bottom: 20px;
        }

        .intro p {
            line-height: 1.5;
            color: #333;
        }

        footer {
            margin-top: 40px;
            padding: 20px 0;
            border-top: 1px solid #e2e2e2;
            text-align: center;
            color: #666;
            font-size: 13px;
        }

        footer a {
            color: #0066cc;
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="header-container">
        <div class="width header">
            <div class="title">
                <h1>Juno Labs PyPI</h1>
            </div>
        </div>
    </div>
    
    <div class="width container">
        <div class="intro">
            <h2>Container-Specific Package Indexes</h2>
            <p>Each index below contains Python packages built for a specific container image. Choose the index that matches your Docker container to install the correct pre-built wheels.</p>
            <p><strong>Usage:</strong> <code>pip install --index-url https://pypi.juno-labs.com/&lt;container-name&gt;/ your-package</code></p>
        </div>
        
        <div class="package-list">
"""
    
    for info in containers_info:
        container_name = info['name']
        title = info['title']
        package_count = info['package_count']
        
        html_content += f"""            <a href="{container_name}/" class="package">
                <strong>{container_name}</strong>
                <span class="package-meta">{package_count} packages</span>
                <br>
                <span style="color: #666; font-size: 12px;">{title}</span>
            </a>
"""
    
    html_content += """        </div>
        
        <footer>
            Built with <a href="https://github.com/chriskuehl/dumb-pypi" target="_blank">dumb-pypi</a> | 
            <a href="https://github.com/juno-ai-labs/pypi" target="_blank">View on GitHub</a>
        </footer>
    </div>
</body>
</html>
"""
    
    output_file = Path(output_dir) / 'index.html'
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Created root index at {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: build_nested_indexes.py <base_url> <output_dir>")
        sys.exit(1)
    
    base_url = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all container directories
    container_dirs = get_container_dirs('packages')
    
    if not container_dirs:
        print("No container directories found in packages/")
        sys.exit(1)
    
    print(f"Found {len(container_dirs)} container directories")
    
    # Build index for each container
    containers_info = []
    for container_name in container_dirs:
        info = build_index_for_container(container_name, base_url, output_dir)
        containers_info.append(info)
    
    # Create root navigation index
    create_root_index(containers_info, output_dir)
    
    print(f"\nSuccessfully built {len(containers_info)} nested indexes")
    print(f"Root index: {output_dir}/index.html")


if __name__ == '__main__':
    main()
