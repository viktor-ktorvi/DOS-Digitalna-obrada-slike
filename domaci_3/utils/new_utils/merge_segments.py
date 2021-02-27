from domaci_3.utils.MyLine import MyLine, make_lines, LineSegment, lines_are_touching, dist


def merge_segments(segments):
    x_max_length = [0, 0]
    y_max_length = [0, 0]
    max_length = 0
    for s1 in segments:
        for s2 in segments:
            length = dist(s1.x_start, s1.y_start, s2.x_end, s2.y_end)
            if length > max_length:
                max_length = length
                x_max_length = [s1.x_start, s2.x_end]
                y_max_length = [s1.y_start, s2.y_end]

    x1 = x_max_length[0]
    x2 = x_max_length[1]

    y1 = y_max_length[0]
    y2 = y_max_length[1]

    return x1, y1, x2, y2
