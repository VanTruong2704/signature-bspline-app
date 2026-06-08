import os
import shutil


def ensure_output_dir(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_temp_image(image, output_dir="output", filename="clipboard_temp.png"):
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    image.save(temp_path)
    return temp_path


def cleanup_temp_dir(output_dir="output"):
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def read_text_file(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
