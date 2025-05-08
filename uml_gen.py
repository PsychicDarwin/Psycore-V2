import subprocess
import os

def get_all_py_files(root):
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".py"):
                py_files.append(os.path.join(dirpath, file))
    return py_files

if __name__ == "__main__":
    project_root = os.getcwd()
    py_files = get_all_py_files(project_root)

    # Optional: Filter out unwanted files (e.g., tests, migrations, etc.)
    py_files = [
        f for f in py_files
        if "__pycache__" not in f and "site-packages" not in f
    ]

    # Run pyreverse with all files
    subprocess.run(
        ["pyreverse", "-o", "png", "-p", "projectuml"] + py_files,
        shell=True  # Required on Windows for correct argument parsing
    )
