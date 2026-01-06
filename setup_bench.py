import subprocess
import sys
import os
import shutil

# ================= CONFIGURATION =================
REPO_URL = "https://github.com/EndeeLabs/VectorDBBench.git"
REPO_DIR = "VectorDBBench"
PYTHON_VERSION = "3.11.9"
# =================================================

def run_command(command, shell=False, cwd=None):
    """Runs a shell command and exits on failure."""
    cmd_str = ' '.join(command) if isinstance(command, list) else command
    print(f"--> [EXEC]: {cmd_str}")
    try:
        subprocess.check_call(command, shell=shell, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def is_python311_installed():
    """Checks if python3.11 is currently available in the system PATH."""
    # Check standard PATH
    if shutil.which("python3.11") is not None:
        return True
    
    # Check common local install location (Source builds often go here)
    if os.path.exists("/usr/local/bin/python3.11"):
        return True
        
    return False

def check_system_compatibility():
    """Ensures we are on a Debian-based system (Debian, Ubuntu, Mint, Kali, etc)."""
    if shutil.which("apt-get") is None:
        print("\n!!! CRITICAL ERROR !!!")
        print("This script relies on 'apt-get'. It works on Debian, Ubuntu, Linux Mint, Kali, Pop!_OS, etc.")
        sys.exit(1)

def is_ubuntu():
    """Returns True only if we are confident this is Ubuntu."""
    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                if "ubuntu" in f.read().lower():
                    return True
        if shutil.which("lsb_release"):
            out = subprocess.check_output(["lsb_release", "-i"]).decode().lower()
            if "ubuntu" in out:
                return True
    except:
        pass
    return False

def install_python_ubuntu_strategy():
    """Strategy A: Ubuntu (Fast PPA)"""
    print("\n[Strategy] Detected Ubuntu. Using fast PPA installation...")
    run_command("sudo apt-get update", shell=True)
    run_command("sudo apt-get install -y software-properties-common git", shell=True)
    print("Adding Deadsnakes PPA...")
    run_command("sudo add-apt-repository -y ppa:deadsnakes/ppa", shell=True)
    run_command("sudo apt-get update", shell=True)
    run_command("sudo apt-get install -y python3.11 python3.11-venv python3.11-dev", shell=True)

def install_python_debian_strategy():
    """Strategy B: Debian / Universal (Source Build Fallback)"""
    print("\n[Strategy] Detected Debian/Other. Using robust compatibility mode...")

    # 1. Install Dependencies
    run_command("sudo apt-get update", shell=True)
    print("Installing build dependencies...")
    deps = [
        "git", "wget", "build-essential", "zlib1g-dev", "libncurses5-dev", 
        "libgdbm-dev", "libnss3-dev", "libssl-dev", "libreadline-dev", 
        "libffi-dev", "libsqlite3-dev", "libbz2-dev", "pkg-config"
    ]
    run_command(f"sudo apt-get install -y {' '.join(deps)}", shell=True)

    # 2. Try APT first (Debian 12+)
    print("Checking system repos for Python 3.11...")
    try:
        subprocess.check_call("apt-cache show python3.11", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Python 3.11 found in APT. Installing...")
        run_command("sudo apt-get install -y python3.11 python3.11-venv python3.11-dev", shell=True)
        return
    except subprocess.CalledProcessError:
        print("Python 3.11 not in system repos. Proceeding to Source Build.")

    # 3. Source Build (Universal)
    print("\n*** STARTING SOURCE BUILD ***")
    tarball = f"Python-{PYTHON_VERSION}.tgz"
    url = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{tarball}"

    if not os.path.exists(tarball):
        run_command(f"wget {url}", shell=True)
    
    run_command(f"tar -xf {tarball}", shell=True)
    src_dir = f"Python-{PYTHON_VERSION}"
    
    # Configure & Make
    run_command("./configure --enable-optimizations", shell=True, cwd=src_dir)
    nproc = subprocess.check_output("nproc", shell=True).decode().strip()
    run_command(f"make -j {nproc}", shell=True, cwd=src_dir)
    
    # SAFE INSTALL (altinstall)
    run_command("sudo make altinstall", shell=True, cwd=src_dir)
    
    # Cleanup
    os.remove(tarball)
    run_command(f"sudo rm -rf {src_dir}", shell=True)

def setup_project_and_venv():
    print("\n[Project] Setting up VectorDBBench...")
    
    # 1. Clone
    if not os.path.exists(REPO_DIR):
        run_command(["git", "clone", REPO_URL])
    
    os.chdir(REPO_DIR)
    
    # 2. Switch Branch
    run_command(["git", "fetch", "origin"])
    run_command(["git", "checkout", "Endee"])
    run_command(["git", "pull", "origin", "Endee"])

    # 3. Locate Python 3.11
    python_bin = "python3.11"
    if shutil.which("python3.11") is None:
        if os.path.exists("/usr/local/bin/python3.11"):
            python_bin = "/usr/local/bin/python3.11"
        else:
            print("Error: Python 3.11 binary not found after installation attempt.")
            sys.exit(1)
            
    print(f"Using Python binary: {python_bin}")

    # 4. Create Venv
    if not os.path.exists("venv"):
        run_command([python_bin, "-m", "venv", "venv"])
    else:
        print("Virtual environment already exists.")

    # 5. Install Deps
    venv_pip = os.path.join("venv", "bin", "pip")
    run_command([venv_pip, "install", "--upgrade", "pip"])
    run_command([venv_pip, "install", "endee"])
    run_command([venv_pip, "install", "-e", "."])
    
    return os.path.join(os.getcwd(), "venv")

if __name__ == "__main__":
    check_system_compatibility()

    # --- THE FIX IS HERE ---
    if is_python311_installed():
        print("\n" + "="*50)
        print("SKIP: Python 3.11 is already installed.")
        print("="*50)
    else:
        # Only install if missing
        if is_ubuntu():
            install_python_ubuntu_strategy()
        else:
            install_python_debian_strategy()
    # -----------------------

    venv_path = setup_project_and_venv()

    print("\n" + "="*50)
    print("SETUP SUCCESSFUL!")
    print(f"To start: source {os.path.join(venv_path, 'bin', 'activate')}")
    print("="*50)





'''
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

'''