import os


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _ensure_parent_dir(file_path):
    """
    Tạo thư mục cha nếu chưa tồn tại.
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _format_point_value(value):
    """
    Format số cho file POINT.
    Theo mẫu POINT đang dùng:
        x y z w
    với 3 chữ số thập phân.
    """
    return f"{float(value):.3f}"


def _format_bspline_value(value):
    """
    Format số cho file BSPLINECURVE.
    Theo mẫu BSPLINECURVE đang dùng:
        x y z w 0
    và knot
    với 8 chữ số thập phân.
    """
    return f"{float(value):.8f}"


def _get_xy(point):
    """
    Lấy (x, y) từ nhiều kiểu dữ liệu điểm.

    Hỗ trợ:
        (x, y)
        [x, y]
        {"x": x, "y": y}
        object có .x và .y
    """
    if isinstance(point, dict):
        return point["x"], point["y"]

    if hasattr(point, "x") and hasattr(point, "y"):
        return point.x, point.y

    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return point[0], point[1]

    raise ValueError(f"Điểm không hợp lệ: {point}")


def _get_curve_value(curve, key, default=None):
    """
    Lấy giá trị từ curve.

    Hỗ trợ:
        curve là dict
        curve là object có thuộc tính
    """
    if isinstance(curve, dict):
        return curve.get(key, default)

    return getattr(curve, key, default)


# =========================================================
# WRITE POINT DAT
# =========================================================

def write_point_dat(strokes, file_path):
    """
    Ghi file diempixel.dat theo định dạng [POINT] của DUTMod/DISCO.

    strokes có dạng:
        [
            [(x1, y1), (x2, y2), ...],
            [(x1, y1), (x2, y2), ...],
            ...
        ]

    Mỗi stroke sẽ được ghi thành 1 block [POINT].
    """
    _ensure_parent_dir(file_path)

    lines = []

    for stroke in strokes:
        if not stroke:
            continue

        lines.append("==========================")
        lines.append("[POINT]")
        lines.append("")
        lines.append(f"{len(stroke)} //numpoint")
        lines.append("0.000 //radius")
        lines.append("0.000  //height")
        lines.append("0.000 //angle degree")
        lines.append("")

        for point in stroke:
            x, y = _get_xy(point)

            z = 0.0
            w = 1.0

            lines.append(
                f"{_format_point_value(x)} "
                f"{_format_point_value(y)} "
                f"{_format_point_value(z)} "
                f"{_format_point_value(w)}"
            )

        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================
# WRITE BSPLINE DAT
# =========================================================

def write_bspline_dat(curves, file_path):
    """
    Ghi file bsplinecurve.dat theo định dạng [BSPLINECURVE] của DUTMod/DISCO.

    curves có dạng:
        [
            {
                "degree": 3,
                "knot_type": 1,
                "control_points": [(x1, y1), (x2, y2), ...],
                "knots": [u0, u1, u2, ...]
            },
            ...
        ]

    Mỗi curve sẽ được ghi thành 1 block [BSPLINECURVE].

    Định dạng block:

        ==========================
        [BSPLINECURVE]

        UNum, UDegree, UKnotType // UNum, UDegree, UKnotType

        // Control Points
        x y z w 0
        ...

        // UKnot
        u0
        u1
        ...
    """
    _ensure_parent_dir(file_path)

    lines = []

    for curve in curves:
        degree = int(_get_curve_value(curve, "degree", default=3))
        knot_type = int(_get_curve_value(curve, "knot_type", default=1))
        control_points = _get_curve_value(curve, "control_points", default=[])
        knots = _get_curve_value(curve, "knots", default=[])

        if not control_points:
            continue

        unum = len(control_points)
        expected_knot_count = unum + degree + 1

        if len(knots) != expected_knot_count:
            raise ValueError(
                "Số lượng knot không đúng định dạng BSPLINECURVE. "
                f"UNum = {unum}, UDegree = {degree}, "
                f"cần {expected_knot_count} knot nhưng nhận được {len(knots)} knot."
            )

        lines.append("==========================")
        lines.append("[BSPLINECURVE]")
        lines.append("")
        lines.append(f"{unum}, {degree}, {knot_type} // UNum, UDegree, UKnotType")
        lines.append("")
        lines.append("// Control Points")

        for point in control_points:
            x, y = _get_xy(point)

            z = 0.0
            w = 1.0

            lines.append(
                f"{_format_bspline_value(x)} "
                f"{_format_bspline_value(y)} "
                f"{_format_bspline_value(z)} "
                f"{_format_bspline_value(w)} "
                f"0"
            )

        lines.append("")
        lines.append("// UKnot")

        for knot in knots:
            lines.append(_format_bspline_value(knot))

        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)