import os
import shutil
from pathlib import Path


def ensure_output_dir(output_dir="output"):
    """
    Tạo thư mục output nếu chưa tồn tại.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_temp_image(image, output_dir="output", filename="clipboard_temp.png"):
    """
    Lưu ảnh tạm từ clipboard vào thư mục temp để xử lý.
    """
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, filename)
    image.save(temp_path)
    return temp_path


def cleanup_temp_dir(output_dir="output"):
    """
    Xoá thư mục temp và các ảnh tạm bên trong sau khi dùng xong.
    """
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def read_text_file(file_path, encoding="utf-8"):
    """
    Đọc nội dung file text.
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(file_path, content, encoding="utf-8"):
    """
    Ghi nội dung chuỗi vào file text.
    Tự tạo thư mục cha nếu chưa tồn tại.
    """
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def is_image_file(file_path):
    """
    Kiểm tra file có phải ảnh thông dụng hay không.
    """
    if not file_path:
        return False

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    suffix = Path(file_path).suffix.lower()

    return suffix in image_extensions


def normalize_path(file_path):
    """
    Chuẩn hoá đường dẫn để hiển thị dễ đọc hơn.
    """
    return os.path.normpath(file_path)


def clamp(value, min_value, max_value):
    """
    Giới hạn một giá trị nằm trong đoạn [min_value, max_value].
    """
    return max(min_value, min(value, max_value))


def distance(p1, p2):
    """
    Tính khoảng cách Euclid giữa 2 điểm 2D.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return (dx * dx + dy * dy) ** 0.5