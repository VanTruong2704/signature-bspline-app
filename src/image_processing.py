import cv2
import numpy as np
import math

# =========================================================
# CONFIG - BẢN ỔN ĐỊNH NÂNG CẤP XỬ LÝ ẢNH THẤP & NỀN GIẤY Màu
# =========================================================

DEFAULT_MAX_IMAGE_SIZE = 950
DEFAULT_MIN_WORK_IMAGE_SIZE = 550      # Kích thước tối thiểu để thuật toán hoạt động tốt
# Xoá viền vật lý an toàn
BORDER_CLEAR_PADDING = 15
FOREGROUND_CROP_PADDING = 28

# Cấu hình lọc cấu trúc gốc
DEFAULT_MIN_COMPONENT_AREA = 4
DEFAULT_SAMPLE_STEP = 2

# Lọc các hạt nhiễu gây ra nét đứt rác
DEFAULT_MIN_STROKE_POINTS = 3
DEFAULT_MIN_STROKE_LENGTH = 2.0

# Khoảng cách nối stroke sau khi đã lấy điểm
ENDPOINT_CONNECT_DISTANCE = 15

# Cấu hình thuật toán nối kết nối xương nối nét đứt cũ
SKELETON_BRIDGE_DISTANCE = 20
SKELETON_BRIDGE_MAX_ITERATIONS = 2
SKELETON_BRIDGE_TRACE_LENGTH = 9
SKELETON_BRIDGE_ANGLE_DOT_LIMIT = 0.45


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def extract_signature_strokes(
    image_path,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    min_work_size=DEFAULT_MIN_WORK_IMAGE_SIZE,   # Bổ sung quản lý ảnh size thấp
    min_component_area=DEFAULT_MIN_COMPONENT_AREA,
    sample_step=DEFAULT_SAMPLE_STEP,
    min_stroke_points=DEFAULT_MIN_STROKE_POINTS,
    min_stroke_length=DEFAULT_MIN_STROKE_LENGTH
):
    image = _read_image(image_path)
    # Tự động điều chỉnh kích thước ảnh thông minh (Thu nhỏ ảnh quá to / Phóng to ảnh quá nhỏ)
    image = _resize_image_keep_ratio(image, max_image_size, min_work_size)
    gray = _to_gray(image)

    # 1. Nhị phân hoá an toàn: Khử hoàn toàn màu giấy, bóng mờ và hỗ trợ chữ ký mực xanh
    binary = _binarize_image(gray)

    # 2. Xoá viền khung vật lý triệt để
    binary = _clear_borders(binary, BORDER_CLEAR_PADDING)

    # 3. Hàn gắn vi mô: chỉ vá vết nứt rất nhỏ, không làm dính chữ quá mạnh
    binary = _heal_cracks(binary)

    # 4. Dọn dẹp rác & cắt viền foreground
    binary = _clean_binary_image(binary, min_component_area)
    binary = _crop_binary_to_foreground(binary, padding=FOREGROUND_CROP_PADDING)

    # 5. Rút khung xương (Skeletonize)
    skeleton = _skeletonize(binary)

    # 6. Nối các đoạn skeleton bị hở đầu mút theo thuật toán ổn định cũ
    skeleton = _bridge_skeleton_endpoint_gaps(
        skeleton,
        max_distance=SKELETON_BRIDGE_DISTANCE,
        max_iterations=SKELETON_BRIDGE_MAX_ITERATIONS,
        trace_length=SKELETON_BRIDGE_TRACE_LENGTH,
        angle_dot_limit=SKELETON_BRIDGE_ANGLE_DOT_LIMIT
    )

    # 7. Trích xuất mảng toạ độ cơ bản từ bộ xương ảnh
    strokes = _extract_strokes_from_skeleton(skeleton, sample_step)

    # 8. Lọc các điểm rác / hạt nhiễu nhỏ lẻ
    strokes = _filter_strokes(strokes, min_stroke_points, min_stroke_length)

    # 9. Nối stroke cấp danh sách điểm (Thuật toán hình học gốc)
    strokes = _connect_strokes_basic(strokes, ENDPOINT_CONNECT_DISTANCE)

    # 10. Đảo toạ độ Y tương thích hệ trục DISCO
    image_height = binary.shape[0]
    strokes = _map_to_disco_coordinates(strokes, image_height)

    return _sort_strokes(strokes)


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _read_image(image_path):
    stream = open(image_path, "rb")
    bytes_arr = bytearray(stream.read())
    numpyarray = np.asarray(bytes_arr, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        white_bg = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_bg.astype(np.float32) * (1 - alpha_factor)
        img = (base + white).astype(np.uint8)
    return img


def _resize_image_keep_ratio(image, max_size, min_size=550):
    h, w = image.shape[:2]
    current_max = max(h, w)

    if current_max > max_size:
        # Thu nhỏ dùng INTER_AREA chống răng cưa hiệu quả nhất
        scale = max_size / current_max
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif current_max < min_size:
        # Phóng to dùng INTER_CUBIC để nội suy mượt viền, làm dày nét chữ tự nhiên
        scale = min_size / current_max
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image


def _to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _binarize_image(gray_image):
    # Bước A: Làm mịn nhẹ để giảm độ nhiễu hạt của thớ giấy tự nhiên
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # Bước B: Ước lượng cấu trúc nền giấy bằng phép giãn nở Morphological Dilation cường độ lớn.
    # Vì chữ ký tối màu còn nền giấy sáng màu, bộ lọc này sẽ tạm thời xóa nét chữ đi để giữ lại phông nền giấy.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.dilate(blurred, kernel)
    
    # Bước C: Trừ ảnh gốc cho ảnh nền giấy ước lượng. 
    # Tất cả những vị trí trùng màu giấy sẽ biến mất (thành màu đen), chỉ giữ lại phần lệch màu (nét chữ ký sẽ sáng bừng lên)
    diff = cv2.absdiff(blurred, background)
    
    # Bước D: Áp dụng phân ngưỡng Otsu trực tiếp trên ảnh hiệu số đã sạch bóng phông nền.
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _clear_borders(binary, padding):
    h, w = binary.shape
    binary[0:padding, :] = 0
    binary[h-padding:h, :] = 0
    binary[:, 0:padding] = 0
    binary[:, w-padding:w] = 0
    return binary


def _heal_cracks(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _clean_binary_image(binary, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def _crop_binary_to_foreground(binary, padding):
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)

    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(binary.shape[1], x + w + padding)
    y_end = min(binary.shape[0], y + h + padding)

    return binary[y_start:y_end, x_start:x_end]


def _skeletonize(binary):
    try:
        return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        skel = np.zeros(binary.shape, np.uint8)
        img = binary.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel


# =========================================================
# BRIDGE SKELETON GAPS
# =========================================================

def _bridge_skeleton_endpoint_gaps(
    skeleton,
    max_distance,
    max_iterations,
    trace_length,
    angle_dot_limit
):
    bridged = skeleton.copy()

    for _ in range(max_iterations):
        endpoints = _find_skeleton_endpoints(bridged)
        if len(endpoints) < 2:
            break

        directions = {
            endpoint: _estimate_endpoint_inner_direction(bridged, endpoint, trace_length)
            for endpoint in endpoints
        }

        pairs = _select_endpoint_pairs_for_bridging(
            endpoints=endpoints,
            directions=directions,
            max_distance=max_distance,
            angle_dot_limit=angle_dot_limit
        )

        if not pairs:
            break

        for p1, p2 in pairs:
            cv2.line(bridged, p1, p2, 255, 1, lineType=cv2.LINE_8)

    return bridged


def _find_skeleton_endpoints(skeleton):
    img = (skeleton > 0).astype(np.uint8)
    h, w = img.shape
    endpoints = []

    ys, xs = np.where(img > 0)
    for x, y in zip(xs, ys):
        x1 = max(0, x - 1)
        x2 = min(w, x + 2)
        y1 = max(0, y - 1)
        y2 = min(h, y + 2)

        neighbor_count = int(np.sum(img[y1:y2, x1:x2])) - 1
        if neighbor_count == 1:
            endpoints.append((int(x), int(y)))

    return endpoints


def _estimate_endpoint_inner_direction(skeleton, endpoint, max_steps):
    img = (skeleton > 0).astype(np.uint8)
    h, w = img.shape

    current = endpoint
    previous = None
    path = [endpoint]

    for _ in range(max_steps):
        x, y = current
        neighbors = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue

                if img[ny, nx] == 0:
                    continue

                candidate = (int(nx), int(ny))
                if previous is not None and candidate == previous:
                    continue

                neighbors.append(candidate)

        if not neighbors:
            break

        next_point = neighbors[0]
        previous = current
        current = next_point
        path.append(current)

        if len(neighbors) > 1:
            break

    if len(path) < 2:
        return None

    sx, sy = path[0]
    ex, ey = path[-1]
    vx = float(ex - sx)
    vy = float(ey - sy)
    length = math.hypot(vx, vy)

    if length <= 1e-9:
        return None

    return (vx / length, vy / length)


def _select_endpoint_pairs_for_bridging(endpoints, directions, max_distance, angle_dot_limit):
    candidates = []

    for i in range(len(endpoints)):
        p1 = endpoints[i]
        d1 = directions.get(p1)

        for j in range(i + 1, len(endpoints)):
            p2 = endpoints[j]
            d2 = directions.get(p2)

            dist = math.dist(p1, p2)
            if dist <= 1.0 or dist > max_distance:
                continue

            if not _is_reasonable_bridge_direction(p1, p2, d1, d2, dist, angle_dot_limit):
                continue

            score = _bridge_score(p1, p2, d1, d2, dist)
            candidates.append((score, p1, p2))

    candidates.sort(key=lambda item: item[0])

    used = set()
    selected = []

    for _, p1, p2 in candidates:
        if p1 in used or p2 in used:
            continue
        selected.append((p1, p2))
        used.add(p1)
        used.add(p2)

    return selected


def _is_reasonable_bridge_direction(p1, p2, d1, d2, dist, angle_dot_limit):
    if dist <= 6.0:
        return True

    if d1 is None or d2 is None:
        return False

    gx = (p2[0] - p1[0]) / dist
    gy = (p2[1] - p1[1]) / dist

    dot1 = d1[0] * gx + d1[1] * gy
    dot2 = d2[0] * (-gx) + d2[1] * (-gy)

    return dot1 <= angle_dot_limit and dot2 <= angle_dot_limit


def _bridge_score(p1, p2, d1, d2, dist):
    if d1 is None or d2 is None:
        return dist

    gx = (p2[0] - p1[0]) / dist
    gy = (p2[1] - p1[1]) / dist

    dot1 = d1[0] * gx + d1[1] * gy
    dot2 = d2[0] * (-gx) + d2[1] * (-gy)

    return dist + max(0.0, dot1) * 8.0 + max(0.0, dot2) * 8.0


# =========================================================
# STROKE EXTRACTION
# =========================================================

def _extract_strokes_from_skeleton(skeleton, sample_step):
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes = []

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) == 0:
            continue

        unique_pts = [pts[0].tolist()]
        for pt in pts[1:]:
            if pt.tolist() != unique_pts[-1]:
                unique_pts.append(pt.tolist())

        if len(unique_pts) < 2:
            continue

        sampled = unique_pts[::sample_step]

        if unique_pts[-1] != sampled[-1]:
            sampled.append(unique_pts[-1])

        strokes.append(sampled)

    return strokes


def _filter_strokes(strokes, min_points, min_length):
    filtered = []
    for stroke in strokes:
        if len(stroke) < min_points:
            continue
        length = sum(math.dist(stroke[i - 1], stroke[i]) for i in range(1, len(stroke)))
        if length >= min_length:
            filtered.append(stroke)
    return filtered


def _connect_strokes_basic(strokes, max_dist):
    if not strokes:
        return []

    merged = []
    used = [False] * len(strokes)

    for i in range(len(strokes)):
        if used[i]:
            continue
        current_stroke = strokes[i].copy()
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(strokes)):
                if used[j]:
                    continue
                candidate = strokes[j]

                d_ss = math.dist(current_stroke[0], candidate[0])
                d_se = math.dist(current_stroke[0], candidate[-1])
                d_es = math.dist(current_stroke[-1], candidate[0])
                d_ee = math.dist(current_stroke[-1], candidate[-1])

                min_d = min(d_ss, d_se, d_es, d_ee)

                if min_d <= max_dist:
                    if min_d == d_es:
                        current_stroke.extend(candidate)
                    elif min_d == d_se:
                        current_stroke = candidate + current_stroke
                    elif min_d == d_ee:
                        current_stroke.extend(candidate[::-1])
                    elif min_d == d_ss:
                        current_stroke = candidate[::-1] + current_stroke

                    used[j] = True
                    changed = True
                    break
        merged.append(current_stroke)
    return merged


def _map_to_disco_coordinates(strokes, height):
    return [[[float(x), float(height - y)] for x, y in stroke] for stroke in strokes]


def _sort_strokes(strokes):
    if not strokes:
        return []
    return sorted(strokes, key=lambda s: s[0][0])
