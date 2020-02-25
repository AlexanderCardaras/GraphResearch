import cv2


def draw_text(img, text_object, size=0.75, weight=2, color=(0, 0, 255)):
    text = text_object[0]
    rect = text_object[1]
    x, y, w, h = rect
    cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight)


def draw_texts(img, text_objects, size=0.75, weight=2, color=(0, 0, 255)):
    for text_object in text_objects:
        draw_text(img, text_object, size, weight, color)


def draw_rect(img, rect_object, weight=1, color=(0, 0, 255)):
    x, y, w, h = rect_object
    cv2.rectangle(img, (x, y), (x+w, y+h), color, weight)


def draw_rects(img, rect_objects, weight=1, color=(0, 0, 255)):
    for rect_object in rect_objects:
        draw_rect(img, rect_object, weight, color)


def draw_point(img, point, weight=3, color=(255, 0, 0)):
    cv2.ellipse(img, (point[0], point[1]), (weight, weight), 0.0, 0.0, 360.0, color, -1)


def draw_corners(img, corners, weight=3, color1=(0, 0, 255), color2=(255, 0, 0)):
    for corner in corners:
        c1 = (corner[0], corner[1])
        c2 = (corner[2], corner[3])
        draw_point(img, c1, weight, color1)
        draw_point(img, c2, weight, color2)


def draw_line(img, line, weight=3, color=(0, 0, 255)):
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, weight)


def draw_lines(img, lines, weight=3, color=(0, 0, 255)):
    for line in lines:
        draw_line(img, line, weight, color)

