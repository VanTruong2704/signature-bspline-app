import os


def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _format_point_value(value):
    return f"{float(value):.3f}"


def _format_bspline_value(value):
    return f"{float(value):.8f}"


def _get_xy(point):
    if isinstance(point, dict):
        return point["x"], point["y"]
    if hasattr(point, "x") and hasattr(point, "y"):
        return point.x, point.y
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return point[0], point[1]
    raise ValueError(f"Điểm không hợp lệ: {point}")


def _get_curve_value(curve, key, default=None):
    return curve.get(key, default) if isinstance(curve, dict) else getattr(curve, key, default)


def write_point_dat(strokes, file_path):
    _ensure_parent_dir(file_path)
    lines = []
    for stroke in strokes:
        if not stroke:
            continue
        lines.extend([
            "==========================",
            "[POINT]",
            "",
            f"{len(stroke)} //numpoint",
            "0.000 //radius",
            "0.000  //height",
            "0.000 //angle degree",
            "",
        ])
        for point in stroke:
            x, y = _get_xy(point)
            lines.append(
                f"{_format_point_value(x)} "
                f"{_format_point_value(y)} "
                f"{_format_point_value(0.0)} "
                f"{_format_point_value(1.0)}"
            )
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_bspline_dat(curves, file_path):
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
        lines.extend([
            "==========================",
            "[BSPLINECURVE]",
            "",
            f"{unum}, {degree}, {knot_type} // UNum, UDegree, UKnotType",
            "",
            "// Control Points",
        ])
        for point in control_points:
            x, y = _get_xy(point)
            lines.append(
                f"{_format_bspline_value(x)} "
                f"{_format_bspline_value(y)} "
                f"{_format_bspline_value(0.0)} "
                f"{_format_bspline_value(1.0)} "
                "0"
            )
        lines.extend(("", "// UKnot"))
        lines.extend(_format_bspline_value(knot) for knot in knots)
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
