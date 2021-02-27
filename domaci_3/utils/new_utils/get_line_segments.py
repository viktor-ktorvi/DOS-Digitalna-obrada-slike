import numpy as np
from domaci_3.utils.array_4d import make_4d_array_custom
from domaci_3.utils.MyLine import LineSegment


def get_line_segments(edges, lines, min_size, max_gaps, tolerancy):
    all_points_j = []
    all_points_i = []
    segments = []
    for line in lines:
        r = tolerancy
        img_padded = np.pad(edges, ((r, r), (r, r)),
                            mode="constant")
        image_4d = make_4d_array_custom(img_padded, edges.shape[0], edges.shape[1], 2 * r + 1,
                                        2 * r + 1)
        points_j = []
        points_i = []
        for j in range(edges.shape[1]):
            y = line.getYatX(j)
            i = round(y)
            if i < 0 or i > edges.shape[0] - 1:
                continue

            if np.count_nonzero(image_4d[i, j]) > 0:
                points_j.append(j)
                points_i.append(i)

        for cnt in range(len(points_j)):
            if cnt == 0:
                seg = LineSegment(x_start=points_j[0], x_end=points_j[0], y_start=points_i[0], y_end=points_i[0])

            if seg.dist2point(points_j[cnt], points_i[cnt]) < max_gaps:
                seg.addPoint(points_j[cnt], points_i[cnt])
            else:
                if seg.length > min_size:
                    segments.append(seg)
                seg = LineSegment(x_start=points_j[cnt], x_end=points_j[cnt], y_start=points_i[cnt],
                                  y_end=points_i[cnt])
        if len(points_j) == 0:
            continue
        if seg.length > min_size:
            segments.append(seg)
        seg = []
        all_points_j = all_points_j + points_j
        all_points_i = all_points_i + points_i

    return segments, all_points_j, all_points_i
