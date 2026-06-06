import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

# Cấu hình ưu tiên giống ảnh gốc nhưng giảm tải CPU.
# 800px thường đủ giữ nét chữ ký, nhẹ hơn 900/1500 đáng kể.
DEFAULT_MAX_IMAGE_SIZE = 800
DEFAULT_MIN_WORK_IMAGE_SIZE = 550
DEFAULT_MAX_UPSCALE_FACTOR = 1.35

# Cắt vùng nền trắng thừa trước khi skeleton để giảm CPU.
FOREGROUND_CROP_PADDING = 28

# Giữ nhiều điểm skeleton để B-spline bám sát nét hơn.
DEFAULT_MIN_COMPONENT_AREA = 6
DEFAULT_SAMPLE_STEP = 1
DEFAULT_MAX_POINTS_PER_STROKE = 1100

# Giữ lại cả các nét/dấu nhỏ trong chữ ký tiếng Việt.
DEFAULT_MIN_STROKE_POINTS = 5
DEFAULT_MIN_STROKE_LENGTH = 6.0

# Nối và gộp path hơi mạnh hơn để giảm đứt đoạn.
# Không nên tăng quá cao vì có thể nối sai dấu/đường gạch vào nét chữ.
ENDPOINT_CONNECT_DISTANCE = 12
PATH_MERGE_DISTANCE = 22
PATH_MERGE_MAX_ROUNDS = 7


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def extract_signature_strokes(
    image_path,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    min_component_area=DEFAULT_MIN_COMPONENT_AREA,
    sample_step=DEFAULT_SAMPLE_STEP,
    max_points_per_stroke=DEFAULT_MAX_POINTS_PER_STROKE
):
    """
    Đọc ảnh chữ ký và trích xuất các nét/cụm điểm thuộc chữ ký.

    Quy trình bản cải tiến:
        1. Đọc ảnh.
        2. Resize nếu ảnh quá lớn.
        3. Chuyển grayscale.
        4. Nhị phân hóa nét chữ ký.
        5. Làm sạch và nối nhẹ nét bị hở.
        6. Làm mảnh nét bằng skeleton.
        7. Nối các endpoint gần nhau trên skeleton.
        8. Trace skeleton thành các path.
        9. Gộp các path gần nhau để giảm rời rạc.
        10. Lấy mẫu và đổi sang tọa độ DISCO.

    Trả về:
        [
            [(x1, y1), (x2, y2), ...],
            [(x1, y1), (x2, y2), ...],
            ...
        ]
    """
    image = _read_image(image_path)
    image = _resize_image_keep_ratio(image, max_image_size)

    gray = _to_gray(image)

    binary = _binarize_signature(gray)
    binary = _clean_binary_image(binary)

    # Cắt bỏ phần nền trắng thừa để skeleton/trace nhẹ hơn và ổn định hơn.
    # Ta dùng kích thước sau khi crop làm hệ tọa độ DISCO, vì DISCO chỉ cần hình dạng nét.
    binary = _crop_binary_to_foreground(binary, padding=FOREGROUND_CROP_PADDING)

    skeleton = _zhang_suen_thinning(binary)

    # Nối các đoạn skeleton bị đứt nhẹ.
    skeleton = _connect_nearby_endpoints(
        skeleton,
        max_distance=ENDPOINT_CONNECT_DISTANCE
    )

    # Không thinning lần 2 để giảm CPU. Đường nối endpoint chỉ dày 1 pixel.

    components = _find_components(
        skeleton,
        min_component_area=min_component_area
    )

    image_height, image_width = skeleton.shape[:2]

    all_paths = []

    for component_mask in components:
        paths = _trace_skeleton_component(component_mask)
        all_paths.extend(paths)

    # Gộp path gần nhau để hạn chế quá nhiều đoạn rời rạc.
    all_paths = _merge_close_paths(
        all_paths,
        max_distance=PATH_MERGE_DISTANCE,
        max_rounds=PATH_MERGE_MAX_ROUNDS
    )

    strokes = []

    for path in all_paths:
        if len(path) < DEFAULT_MIN_STROKE_POINTS:
            continue

        path_length = _polyline_length(path)

        if path_length < DEFAULT_MIN_STROKE_LENGTH:
            continue

        sampled_path = _resample_and_limit_path(
            path,
            sample_step=sample_step,
            max_points=max_points_per_stroke
        )

        if len(sampled_path) < DEFAULT_MIN_STROKE_POINTS:
            continue

        stroke = _pixels_to_disco_points(
            sampled_path,
            image_width=image_width,
            image_height=image_height
        )

        strokes.append(stroke)

    if not strokes:
        strokes = _fallback_contour_strokes(
            binary,
            image_width=image_width,
            image_height=image_height,
            sample_step=sample_step,
            max_points=max_points_per_stroke,
            min_component_area=min_component_area
        )

    strokes = _sort_strokes(strokes)

    return strokes


# =========================================================
# IMAGE READING / BASIC PROCESSING
# =========================================================

def _read_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return image


def _resize_image_keep_ratio(image, max_size):
    """
    Chuẩn hoá kích thước ảnh trước khi xử lý.

    Lý do:
        - Ảnh quá lớn: giảm xuống để thuật toán ổn định và file .dat không quá nặng.
        - Ảnh quá nhỏ: chỉ phóng nhẹ, không upscale quá nhiều vì dễ sinh nét giả.

    Quy tắc hiện tại:
        - Cạnh dài > max_size: resize xuống max_size.
        - Cạnh dài < DEFAULT_MIN_WORK_IMAGE_SIZE: phóng nhẹ lên, tối đa DEFAULT_MAX_UPSCALE_FACTOR.
        - Còn lại: giữ nguyên.
    """
    height, width = image.shape[:2]
    longest_side = max(width, height)

    if longest_side <= 0:
        return image

    if longest_side > max_size:
        scale = max_size / longest_side
        interpolation = cv2.INTER_AREA
    elif longest_side < DEFAULT_MIN_WORK_IMAGE_SIZE:
        scale_to_min = DEFAULT_MIN_WORK_IMAGE_SIZE / longest_side
        scale = min(scale_to_min, DEFAULT_MAX_UPSCALE_FACTOR)
        interpolation = cv2.INTER_CUBIC
    else:
        return image

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )


def _to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarize_signature(gray):
    """
    Nhị phân hóa ảnh chữ ký.

    Mục tiêu:
        nét chữ ký = 255
        nền = 0

    Bản này ưu tiên Otsu vì ảnh chữ ký thường là nền sáng, nét tối.
    Adaptive threshold chỉ dùng bổ sung khi Otsu lấy quá ít nét.
    """
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground_ratio = np.count_nonzero(binary_otsu) / binary_otsu.size

    # Nếu Otsu lấy quá ít nét, dùng adaptive để bổ sung.
    if foreground_ratio < 0.005:
        binary_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9
        )

        binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
    else:
        binary = binary_otsu

    return binary


def _clean_binary_image(binary):
    """
    Làm sạch ảnh nhị phân trước khi skeleton.

    Khác bản trước:
        - Không dùng MORPH_OPEN mạnh vì dễ làm đứt nét mảnh.
        - Dùng CLOSE để nối khe hở nhỏ.
        - Dùng DILATE rất nhẹ nếu nét quá mảnh.
        - Lọc component nhỏ sau cùng.
    """
    cleaned = binary.copy()

    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Nối thêm các khe hở nhỏ theo chiều ngang/dọc.
    kernel_horizontal = np.ones((1, 3), np.uint8)
    kernel_vertical = np.ones((3, 1), np.uint8)

    horizontal = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_horizontal, iterations=1)
    vertical = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_vertical, iterations=1)

    # Nối nhẹ thêm theo hai hướng chéo để giảm đứt các nét nghiêng.
    kernel_diag_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    kernel_diag_2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    diagonal_1 = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_diag_1, iterations=1)
    diagonal_2 = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_diag_2, iterations=1)

    cleaned = cv2.bitwise_or(cleaned, horizontal)
    cleaned = cv2.bitwise_or(cleaned, vertical)
    cleaned = cv2.bitwise_or(cleaned, diagonal_1)
    cleaned = cv2.bitwise_or(cleaned, diagonal_2)

    cleaned = _remove_small_components(
        cleaned,
        min_area=DEFAULT_MIN_COMPONENT_AREA
    )

    return cleaned


def _crop_binary_to_foreground(binary, padding=FOREGROUND_CROP_PADDING):
    """
    Cắt vùng nền trắng thừa quanh chữ ký.

    Việc này giúp giảm rất nhiều số pixel phải skeleton hóa,
    nên CPU mát hơn và app ít bị treo hơn.
    """
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        return binary

    height, width = binary.shape[:2]

    x_min = max(0, int(xs.min()) - padding)
    x_max = min(width, int(xs.max()) + padding + 1)
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(height, int(ys.max()) + padding + 1)

    cropped = binary[y_min:y_max, x_min:x_max]

    if cropped.size == 0:
        return binary

    return cropped


def _remove_small_components(binary, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    result = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            result[labels == label] = 255

    return result


# =========================================================
# SKELETONIZATION
# =========================================================

def _zhang_suen_thinning(binary):
    """
    Làm mảnh nét chữ ký bằng thuật toán Zhang-Suen.

    Input:
        binary:
            ảnh nhị phân, nét chữ ký = 255, nền = 0

    Output:
        skeleton:
            nét mảnh 1 pixel = 255, nền = 0
    """
    img = (binary > 0).astype(np.uint8)

    changed = True
    iteration_count = 0
    max_iterations = 160

    while changed and iteration_count < max_iterations:
        changed = False
        iteration_count += 1

        marker = _zhang_suen_iteration(img, step=1)

        if np.any(marker):
            img[marker] = 0
            changed = True

        marker = _zhang_suen_iteration(img, step=2)

        if np.any(marker):
            img[marker] = 0
            changed = True

    return (img * 255).astype(np.uint8)


def _zhang_suen_iteration(img, step):
    p2 = np.roll(img, 1, axis=0)
    p3 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
    p4 = np.roll(img, -1, axis=1)
    p5 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
    p6 = np.roll(img, -1, axis=0)
    p7 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
    p8 = np.roll(img, 1, axis=1)
    p9 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)

    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

    neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

    transitions = np.zeros_like(img, dtype=np.uint8)

    for i in range(8):
        current_neighbor = neighbors[i]
        next_neighbor = neighbors[(i + 1) % 8]

        transitions += (
            (current_neighbor == 0) &
            (next_neighbor == 1)
        ).astype(np.uint8)

    condition_basic = (
        (img == 1) &
        (neighbor_count >= 2) &
        (neighbor_count <= 6) &
        (transitions == 1)
    )

    if step == 1:
        condition_step = (
            (p2 * p4 * p6 == 0) &
            (p4 * p6 * p8 == 0)
        )
    else:
        condition_step = (
            (p2 * p4 * p8 == 0) &
            (p2 * p6 * p8 == 0)
        )

    marker = condition_basic & condition_step

    marker[0, :] = False
    marker[-1, :] = False
    marker[:, 0] = False
    marker[:, -1] = False

    return marker


# =========================================================
# ENDPOINT CONNECTION
# =========================================================

def _connect_nearby_endpoints(skeleton, max_distance):
    """
    Nối các endpoint gần nhau trên skeleton.

    Mục tiêu:
        Nếu nét chữ ký bị đứt một đoạn nhỏ do threshold/skeleton,
        ta nối lại để B-spline không bị chia thành quá nhiều curve nhỏ.

    Cách làm:
        - Tìm endpoint: pixel skeleton có đúng 1 neighbor.
        - Tìm cặp endpoint gần nhau.
        - Chỉ nối nếu khoảng cách đủ nhỏ.
        - Vẽ line 1 pixel nối chúng.
    """
    result = skeleton.copy()

    endpoints = _find_endpoints(result)

    if len(endpoints) < 2:
        return result

    used = set()
    pairs = []

    max_distance_sq = max_distance * max_distance

    for i in range(len(endpoints)):
        p1 = endpoints[i]

        if p1 in used:
            continue

        best_p2 = None
        best_distance_sq = None

        for j in range(i + 1, len(endpoints)):
            p2 = endpoints[j]

            if p2 in used:
                continue

            distance_sq = _distance_sq(p1, p2)

            if distance_sq > max_distance_sq:
                continue

            if best_distance_sq is None or distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_p2 = p2

        if best_p2 is not None:
            pairs.append((p1, best_p2))
            used.add(p1)
            used.add(best_p2)

    for p1, p2 in pairs:
        cv2.line(
            result,
            p1,
            p2,
            color=255,
            thickness=1
        )

    return result


def _find_endpoints(skeleton):
    points = _mask_to_point_set(skeleton)
    neighbor_map = _build_neighbor_map(points)

    endpoints = []

    for point, neighbors in neighbor_map.items():
        if len(neighbors) == 1:
            endpoints.append(point)

    endpoints.sort(key=lambda p: (p[1], p[0]))

    return endpoints


# =========================================================
# COMPONENTS
# =========================================================

def _find_components(binary, min_component_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_component_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        mask = np.zeros_like(binary)
        mask[labels == label] = 255

        components.append({
            "mask": mask,
            "x": x,
            "y": y,
            "area": area
        })

    components.sort(key=lambda item: (item["y"], item["x"]))

    return [item["mask"] for item in components]


# =========================================================
# SKELETON TRACING
# =========================================================

def _trace_skeleton_component(component_mask):
    """
    Trace một component skeleton thành nhiều path.

    Skeleton được xem như graph pixel:
        pixel = node
        neighbor 8 hướng = edge
    """
    points = _mask_to_point_set(component_mask)

    if not points:
        return []

    neighbor_map = _build_neighbor_map(points)
    node_points = _find_graph_nodes(neighbor_map)

    paths = []
    visited_edges = set()

    if node_points:
        ordered_starts = _order_start_nodes(node_points, neighbor_map)

        for start in ordered_starts:
            for neighbor in _sorted_neighbors(start, neighbor_map[start]):
                edge_key = _edge_key(start, neighbor)

                if edge_key in visited_edges:
                    continue

                path = _trace_from_node(
                    start=start,
                    next_point=neighbor,
                    node_points=node_points,
                    neighbor_map=neighbor_map,
                    visited_edges=visited_edges
                )

                if len(path) >= DEFAULT_MIN_STROKE_POINTS:
                    paths.append(path)

    remaining_paths = _trace_remaining_edges(
        points=points,
        neighbor_map=neighbor_map,
        visited_edges=visited_edges
    )

    for path in remaining_paths:
        if len(path) >= DEFAULT_MIN_STROKE_POINTS:
            paths.append(path)

    paths = _remove_duplicate_or_tiny_paths(paths)

    return paths


def _mask_to_point_set(mask):
    ys, xs = np.where(mask > 0)

    points = set()

    for x, y in zip(xs, ys):
        points.add((int(x), int(y)))

    return points


def _build_neighbor_map(points):
    neighbor_map = {}

    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    for x, y in points:
        neighbors = []

        for dx, dy in directions:
            candidate = (x + dx, y + dy)

            if candidate in points:
                neighbors.append(candidate)

        neighbor_map[(x, y)] = neighbors

    return neighbor_map


def _find_graph_nodes(neighbor_map):
    """
    Node đặc biệt:
        endpoint: degree = 1
        junction: degree >= 3
        isolated: degree = 0

    Pixel degree = 2 là pixel nằm giữa path.
    """
    nodes = set()

    for point, neighbors in neighbor_map.items():
        degree = len(neighbors)

        if degree != 2:
            nodes.add(point)

    return nodes


def _order_start_nodes(node_points, neighbor_map):
    endpoints = []
    junctions = []
    isolated = []

    for point in node_points:
        degree = len(neighbor_map[point])

        if degree == 0:
            isolated.append(point)
        elif degree == 1:
            endpoints.append(point)
        else:
            junctions.append(point)

    endpoints.sort(key=lambda p: (p[1], p[0]))
    junctions.sort(key=lambda p: (p[1], p[0]))
    isolated.sort(key=lambda p: (p[1], p[0]))

    return endpoints + junctions + isolated


def _sorted_neighbors(point, neighbors):
    x, y = point

    return sorted(
        neighbors,
        key=lambda p: (
            (p[1] - y) * (p[1] - y) + (p[0] - x) * (p[0] - x),
            p[1],
            p[0]
        )
    )


def _trace_from_node(
    start,
    next_point,
    node_points,
    neighbor_map,
    visited_edges
):
    path = [start, next_point]

    visited_edges.add(_edge_key(start, next_point))

    previous = start
    current = next_point

    while current not in node_points:
        candidates = [
            p for p in neighbor_map[current]
            if p != previous
        ]

        if not candidates:
            break

        next_candidates = [
            p for p in candidates
            if _edge_key(current, p) not in visited_edges
        ]

        if not next_candidates:
            break

        chosen = _choose_next_point(
            previous=previous,
            current=current,
            candidates=next_candidates
        )

        visited_edges.add(_edge_key(current, chosen))
        path.append(chosen)

        previous = current
        current = chosen

    return path


def _trace_remaining_edges(points, neighbor_map, visited_edges):
    paths = []

    for point in sorted(points, key=lambda p: (p[1], p[0])):
        for neighbor in _sorted_neighbors(point, neighbor_map[point]):
            edge_key = _edge_key(point, neighbor)

            if edge_key in visited_edges:
                continue

            path = [point, neighbor]
            visited_edges.add(edge_key)

            previous = point
            current = neighbor

            while True:
                candidates = [
                    p for p in neighbor_map[current]
                    if p != previous and _edge_key(current, p) not in visited_edges
                ]

                if not candidates:
                    break

                chosen = _choose_next_point(
                    previous=previous,
                    current=current,
                    candidates=candidates
                )

                visited_edges.add(_edge_key(current, chosen))
                path.append(chosen)

                previous = current
                current = chosen

                if current == point:
                    break

            if len(path) >= DEFAULT_MIN_STROKE_POINTS:
                paths.append(path)

    return paths


def _choose_next_point(previous, current, candidates):
    """
    Chọn hướng đi tiếp sao cho ít gãy nhất.
    """
    if len(candidates) == 1:
        return candidates[0]

    px, py = previous
    cx, cy = current

    vx = cx - px
    vy = cy - py

    best_point = candidates[0]
    best_score = None

    for candidate in candidates:
        nx, ny = candidate

        wx = nx - cx
        wy = ny - cy

        dot = vx * wx + vy * wy
        norm_v = (vx * vx + vy * vy) ** 0.5
        norm_w = (wx * wx + wy * wy) ** 0.5

        if norm_v <= 1e-12 or norm_w <= 1e-12:
            score = -1.0
        else:
            score = dot / (norm_v * norm_w)

        if best_score is None or score > best_score:
            best_score = score
            best_point = candidate

    return best_point


def _edge_key(p1, p2):
    return tuple(sorted([p1, p2]))


def _remove_duplicate_or_tiny_paths(paths):
    result = []
    seen = set()

    for path in paths:
        if len(path) < DEFAULT_MIN_STROKE_POINTS:
            continue

        key = (path[0], path[-1], len(path))
        reverse_key = (path[-1], path[0], len(path))

        if key in seen or reverse_key in seen:
            continue

        seen.add(key)
        result.append(path)

    return result


# =========================================================
# PATH MERGING
# =========================================================

def _merge_close_paths(paths, max_distance, max_rounds):
    """
    Gộp các path có đầu/cuối gần nhau.

    Mục tiêu:
        Giảm số lượng curve nhỏ bị rời rạc.

    Hàm này gộp theo 4 trường hợp:
        end(A) gần start(B)
        start(A) gần end(B)
        start(A) gần start(B)
        end(A) gần end(B)
    """
    paths = [p for p in paths if len(p) >= DEFAULT_MIN_STROKE_POINTS]

    if len(paths) <= 1:
        return paths

    for _ in range(max_rounds):
        merged_any = False
        used = [False] * len(paths)
        new_paths = []

        for i in range(len(paths)):
            if used[i]:
                continue

            current = paths[i]
            used[i] = True

            best_j = None
            best_case = None
            best_distance = None

            for j in range(len(paths)):
                if i == j or used[j]:
                    continue

                candidate = paths[j]

                merge_case, distance = _get_best_merge_case(
                    current,
                    candidate,
                    max_distance=max_distance
                )

                if merge_case is None:
                    continue

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_case = merge_case
                    best_j = j

            if best_j is not None:
                current = _merge_two_paths(
                    current,
                    paths[best_j],
                    best_case
                )
                used[best_j] = True
                merged_any = True

            new_paths.append(current)

        paths = new_paths

        if not merged_any:
            break

    return paths


def _get_best_merge_case(path_a, path_b, max_distance):
    a_start = path_a[0]
    a_end = path_a[-1]
    b_start = path_b[0]
    b_end = path_b[-1]

    cases = [
        ("a_end_b_start", _distance(a_end, b_start)),
        ("a_start_b_end", _distance(a_start, b_end)),
        ("a_start_b_start", _distance(a_start, b_start)),
        ("a_end_b_end", _distance(a_end, b_end)),
    ]

    cases.sort(key=lambda item: item[1])

    best_case, best_distance = cases[0]

    if best_distance > max_distance:
        return None, None

    return best_case, best_distance


def _merge_two_paths(path_a, path_b, merge_case):
    if merge_case == "a_end_b_start":
        connector = _line_points(path_a[-1], path_b[0])
        return path_a + connector[1:-1] + path_b

    if merge_case == "a_start_b_end":
        connector = _line_points(path_b[-1], path_a[0])
        return path_b + connector[1:-1] + path_a

    if merge_case == "a_start_b_start":
        path_b_reversed = list(reversed(path_b))
        connector = _line_points(path_b_reversed[-1], path_a[0])
        return path_b_reversed + connector[1:-1] + path_a

    if merge_case == "a_end_b_end":
        path_b_reversed = list(reversed(path_b))
        connector = _line_points(path_a[-1], path_b_reversed[0])
        return path_a + connector[1:-1] + path_b_reversed

    return path_a


def _line_points(p1, p2):
    """
    Sinh các điểm trên đoạn thẳng nối p1-p2.
    Dùng để chèn vào giữa hai path khi gộp.
    """
    x1, y1 = p1
    x2, y2 = p2

    length = int(max(abs(x2 - x1), abs(y2 - y1)))

    if length <= 0:
        return [p1]

    points = []

    for i in range(length + 1):
        t = i / length
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        points.append((x, y))

    return points


# =========================================================
# SAMPLING / GEOMETRY
# =========================================================

def _resample_and_limit_path(path, sample_step, max_points):
    if not path:
        return []

    path = _remove_consecutive_duplicates(path)

    if len(path) <= 2:
        return path

    min_distance = max(1.0, float(sample_step))

    sampled = _sample_by_distance(path, min_distance=min_distance)

    if len(sampled) > max_points:
        sampled = _limit_points_evenly(sampled, max_points=max_points)

    return sampled


def _remove_consecutive_duplicates(points):
    if not points:
        return []

    result = [points[0]]

    for point in points[1:]:
        if point != result[-1]:
            result.append(point)

    return result


def _sample_by_distance(points, min_distance):
    if not points:
        return []

    result = [points[0]]
    last = points[0]

    min_distance_sq = min_distance * min_distance

    for point in points[1:-1]:
        dx = point[0] - last[0]
        dy = point[1] - last[1]

        if dx * dx + dy * dy >= min_distance_sq:
            result.append(point)
            last = point

    if points[-1] != result[-1]:
        result.append(points[-1])

    return result


def _limit_points_evenly(points, max_points):
    if len(points) <= max_points:
        return points

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
        dtype=int
    )

    return [points[i] for i in indices]


def _polyline_length(points):
    if len(points) < 2:
        return 0.0

    total = 0.0

    for i in range(1, len(points)):
        total += _distance(points[i - 1], points[i])

    return total


def _pixels_to_disco_points(pixels, image_width, image_height):
    center_x = image_width / 2.0
    center_y = image_height / 2.0

    points = []

    for x, y in pixels:
        new_x = x - center_x
        new_y = center_y - y

        points.append((float(new_x), float(new_y)))

    return points


def _sort_strokes(strokes):
    def stroke_key(stroke):
        min_y = min(p[1] for p in stroke)
        min_x = min(p[0] for p in stroke)
        return (-min_y, min_x)

    return sorted(strokes, key=stroke_key)


def _distance(p1, p2):
    return _distance_sq(p1, p2) ** 0.5


def _distance_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return dx * dx + dy * dy


# =========================================================
# FALLBACK CONTOUR METHOD
# =========================================================

def _fallback_contour_strokes(
    binary,
    image_width,
    image_height,
    sample_step,
    max_points,
    min_component_area
):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    strokes = []

    for contour in contours:
        if cv2.contourArea(contour) < min_component_area:
            continue

        pixels = []

        for item in contour:
            x = int(item[0][0])
            y = int(item[0][1])
            pixels.append((x, y))

        pixels = _resample_and_limit_path(
            pixels,
            sample_step=sample_step,
            max_points=max_points
        )

        if len(pixels) < DEFAULT_MIN_STROKE_POINTS:
            continue

        stroke = _pixels_to_disco_points(
            pixels,
            image_width=image_width,
            image_height=image_height
        )

        strokes.append(stroke)

    return _sort_strokes(strokes)


# =========================================================
# DEBUG HELPERS
# =========================================================

def save_debug_binary_image(image_path, output_path="output/debug_binary.png"):
    image = _read_image(image_path)
    image = _resize_image_keep_ratio(image, DEFAULT_MAX_IMAGE_SIZE)

    gray = _to_gray(image)
    binary = _binarize_signature(gray)
    binary = _clean_binary_image(binary)
    binary = _crop_binary_to_foreground(binary, padding=FOREGROUND_CROP_PADDING)

    cv2.imwrite(output_path, binary)


def save_debug_skeleton_image(image_path, output_path="output/debug_skeleton.png"):
    image = _read_image(image_path)
    image = _resize_image_keep_ratio(image, DEFAULT_MAX_IMAGE_SIZE)

    gray = _to_gray(image)
    binary = _binarize_signature(gray)
    binary = _clean_binary_image(binary)

    # Cắt bỏ phần nền trắng thừa để skeleton/trace nhẹ hơn và ổn định hơn.
    # Ta dùng kích thước sau khi crop làm hệ tọa độ DISCO, vì DISCO chỉ cần hình dạng nét.
    binary = _crop_binary_to_foreground(binary, padding=FOREGROUND_CROP_PADDING)

    skeleton = _zhang_suen_thinning(binary)
    skeleton = _connect_nearby_endpoints(
        skeleton,
        max_distance=ENDPOINT_CONNECT_DISTANCE
    )
    skeleton = _zhang_suen_thinning(skeleton)

    cv2.imwrite(output_path, skeleton)