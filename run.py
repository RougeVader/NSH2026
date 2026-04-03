import subprocess
import sys
import os

def run_script(script_name):
    # Get the absolute path of the current directory (project root)
    root_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Set up the environment with PYTHONPATH pointing to root
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"--- TeamAtomV2: Running {script_name} ---")
    try:
        # Run the requested script
        subprocess.run([sys.executable, script_name], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script_name>.py")
        print("Example: python run.py acm/main.py")
        sys.exit(1)
    
    run_script(sys.argv[1])
