import subprocess
import sys
import os
import shutil

def run_command(command, shell=False):
    """Runs a shell command and raises an exception on failure."""
    print(f"--> Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        subprocess.check_call(command, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def check_and_install_system_deps():
    print("### Checking System Dependencies ###")
    
    # 1. Update Apt
    run_command("sudo apt-get update", shell=True)

    # 2. Check/Install Git
    if shutil.which("git") is None:
        print("Git not found. Installing...")
        run_command("sudo apt-get install -y git", shell=True)
    else:
        print("Git is already installed.")

    # 3. Check/Install Python 3.11
    # We explicitly check for a python3.11 binary
    if shutil.which("python3.11") is None:
        print("Python 3.11 not found. Installing via deadsnakes PPA...")
        run_command("sudo apt-get install -y software-properties-common", shell=True)
        run_command("sudo add-apt-repository -y ppa:deadsnakes/ppa", shell=True)
        run_command("sudo apt-get update", shell=True)
        run_command("sudo apt-get install -y python3.11 python3.11-venv python3.11-dev", shell=True)
    else:
        print("Python 3.11 is already installed.")

def setup_repo():
    repo_url = "https://github.com/EndeeLabs/VectorDBBench.git"
    repo_dir = "VectorDBBench"

    print("\n### Setting up Repository ###")

    # 1. Clone
    if not os.path.exists(repo_dir):
        run_command(["git", "clone", repo_url])
    else:
        print(f"Directory {repo_dir} already exists. Skipping clone.")

    # Change working directory to the repo
    os.chdir(repo_dir)

    # 2. Switch Branch
    print("Switching to 'Endee' branch...")
    run_command(["git", "fetch", "origin"])
    run_command(["git", "checkout", "Endee"])
    
    # Pull latest just in case
    run_command(["git", "pull", "origin", "Endee"])

    return os.getcwd()

def setup_venv_and_install(repo_path):
    print("\n### Setting up Virtual Environment & Installing Packages ###")
    
    venv_dir = os.path.join(repo_path, "venv")
    
    # 1. Create Venv using Python 3.11 specifically
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run_command(["python3.11", "-m", "venv", "venv"])
    else:
        print("Virtual environment already exists.")

    # Define paths to the venv executables
    venv_python = os.path.join(venv_dir, "bin", "python")
    venv_pip = os.path.join(venv_dir, "bin", "pip")

    # Upgrade pip first
    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    # 2. pip install endee
    print("Installing 'endee'...")
    run_command([venv_pip, "install", "endee"])

    # 3. pip install -e .
    print("Installing editable project...")
    run_command([venv_pip, "install", "-e", "."])

    return venv_dir

if __name__ == "__main__":
    # Ensure we are on Linux (quick check)
    if not sys.platform.startswith('linux'):
        print("Warning: This script is optimized for Linux (AWS/GCP instances).")

    # Step 1: System Level Installs (Sudo required)
    check_and_install_system_deps()

    # Step 2: Clone and Switch Branch
    project_path = setup_repo()

    # Step 3 & 4: Create Venv and Install Pip packages
    venv_path = setup_venv_and_install(project_path)

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("To start using VectorDBBench, run the following command in your shell:")
    print(f"\n    source {os.path.join(project_path, 'venv/bin/activate')}\n")
    print("="*50)