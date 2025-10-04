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
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Juno Labs PyPI - Container Images</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        header p {
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        .content {
            padding: 3rem 2rem;
        }
        
        .intro {
            margin-bottom: 3rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .intro h2 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        
        .intro p {
            color: #555;
            margin-bottom: 0.5rem;
        }
        
        .intro code {
            background: #e9ecef;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #d63384;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .card {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 2rem;
            transition: all 0.3s ease;
            text-decoration: none;
            color: inherit;
            display: block;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.3rem;
            word-break: break-word;
        }
        
        .card-meta {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        .card-meta .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .card-description {
            color: #555;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        .card-footer {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #e9ecef;
            color: #667eea;
            font-weight: 600;
            display: flex;
            align-items: center;
        }
        
        .card-footer::after {
            content: '→';
            margin-left: auto;
            font-size: 1.5rem;
        }
        
        footer {
            background: #f8f9fa;
            padding: 2rem;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        
        footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }
            
            header h1 {
                font-size: 2rem;
            }
            
            .content {
                padding: 2rem 1rem;
            }
            
            .grid {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐍 Juno Labs PyPI</h1>
            <p>Private Python Package Index - Container-Specific Builds</p>
        </header>
        
        <div class="content">
            <div class="intro">
                <h2>📦 Container-Specific Package Indexes</h2>
                <p>Each index below contains Python packages built for a specific container image. Choose the index that matches your Docker container to install the correct pre-built wheels.</p>
                <p><strong>Usage:</strong> <code>pip install --index-url https://juno-ai-labs.github.io/pypi/&lt;container-name&gt;/ your-package</code></p>
            </div>
            
            <div class="grid">
"""
    
    for info in containers_info:
        container_name = info['name']
        title = info['title']
        package_count = info['package_count']
        
        html_content += f"""                <a href="{container_name}/" class="card">
                    <h3>{container_name}</h3>
                    <div class="card-meta">
                        <span class="badge">{package_count} packages</span>
                    </div>
                    <div class="card-description">
                        {title}
                    </div>
                    <div class="card-footer">
                        View package index
                    </div>
                </a>
"""
    
    html_content += """            </div>
        </div>
        
        <footer>
            <p>Built with <a href="https://github.com/chriskuehl/dumb-pypi" target="_blank">dumb-pypi</a> | 
            <a href="https://github.com/juno-ai-labs/pypi" target="_blank">View on GitHub</a></p>
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
