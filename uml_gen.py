import subprocess
import os
from pathlib import Path
from PIL import Image

EXCLUDED_DIRS = {'.venv', 'venv', '__pycache__'}

def find_python_modules(root: Path):
    modules = []
    for path in root.rglob("__init__.py"):
        if not any(part in EXCLUDED_DIRS for part in path.parts):
            modules.append(str(path.relative_to(root)))
    return modules

def generate_uml_diagrams(modules, output_dir="uml_output", image_format="png"):
    os.makedirs(output_dir, exist_ok=True)

    for mod in modules:
        mod_path = str(Path(mod).parent)
        if not mod_path:
            continue  # skip root-level files

        package_name = mod_path.replace(os.sep, "_")

        print(f"Generating UML for: {mod_path}")
        subprocess.run(
            ["pyreverse", "-o", image_format, "-p", package_name, mod_path],
            cwd=Path.cwd(),
            shell=True
        )

        for filetype in ["classes", "packages"]:
            file = f"{filetype}_{package_name}.{image_format}"
            if Path(file).exists():
                Path(file).rename(Path(output_dir) / file)

def stitch_images(image_dir="uml_output", output_file="combined_uml.png", layout="vertical"):
    image_paths = sorted(Path(image_dir).glob("*.png"))
    if not image_paths:
        print("No images found to stitch.")
        return

    images = [Image.open(p) for p in image_paths]

    # Determine size of the final image
    if layout == "vertical":
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
    else:
        total_width = sum(img.width for img in images)
        total_height = max(img.height for img in images)

    combined = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    offset = 0
    for img in images:
        if layout == "vertical":
            combined.paste(img, (0, offset))
            offset += img.height
        else:
            combined.paste(img, (offset, 0))
            offset += img.width

    combined.save(output_file)
    print(f"✅ Combined UML image saved to: {output_file}")

if __name__ == "__main__":
    project_root = Path.cwd()
    init_files = find_python_modules(project_root)

    if not init_files:
        print("❌ No Python packages found. Make sure your project has __init__.py files.")
    else:
        generate_uml_diagrams(init_files)
        stitch_images()
