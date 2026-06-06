import os
from pathlib import Path


def ensure_output_dir(output_dir="output"):
    """
    Tạo thư mục output nếu chưa tồn tại.

    Tham số:
        output_dir: tên hoặc đường dẫn thư mục cần tạo.

    Trả về:
        đường dẫn thư mục output dạng string.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def read_text_file(file_path, encoding="utf-8"):
    """
    Đọc nội dung file text.

    Tham số:
        file_path: đường dẫn file cần đọc.
        encoding: bảng mã file, mặc định utf-8.

    Trả về:
        nội dung file dạng chuỗi.
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(file_path, content, encoding="utf-8"):
    """
    Ghi nội dung chuỗi vào file text.
    Tự tạo thư mục cha nếu chưa tồn tại.

    Tham số:
        file_path: đường dẫn file cần ghi.
        content: nội dung cần ghi.
        encoding: bảng mã file, mặc định utf-8.
    """
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def is_image_file(file_path):
    """
    Kiểm tra file có phải ảnh thông dụng hay không.

    Hỗ trợ:
        .png, .jpg, .jpeg, .bmp

    Trả về:
        True nếu là file ảnh hợp lệ, ngược lại False.
    """
    if not file_path:
        return False

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    suffix = Path(file_path).suffix.lower()

    return suffix in image_extensions


def normalize_path(file_path):
    """
    Chuẩn hoá đường dẫn để hiển thị dễ đọc hơn.

    Ví dụ:
        output\\diempixel.dat
        output/diempixel.dat

    Trả về:
        đường dẫn dạng string.
    """
    return os.path.normpath(file_path)


def clamp(value, min_value, max_value):
    """
    Giới hạn một giá trị nằm trong đoạn [min_value, max_value].

    Ví dụ:
        clamp(15, 0, 10) -> 10
        clamp(-2, 0, 10) -> 0
        clamp(5, 0, 10) -> 5
    """
    return max(min_value, min(value, max_value))


def distance(p1, p2):
    """
    Tính khoảng cách Euclid giữa 2 điểm 2D.

    p1, p2 có dạng:
        (x, y)

    Trả về:
        khoảng cách dạng float.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return (dx * dx + dy * dy) ** 0.5