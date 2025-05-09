import subprocess
import os

if __name__ == "__main__":
    project_root = os.getcwd()

    # Put your actual package/module folder names here
    modules_to_diagram = ["src"]  # <-- update this

    for mod in modules_to_diagram:
        subprocess.run(
            ["pyreverse", "-o", "png", "-p", mod, mod],
            cwd=project_root,
            shell=True
        )