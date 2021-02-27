import cv2


def segment_lanes(img):
    white = (255, 255, 245)
    whiteish = (240, 220, 210)
    yellow = (200, 170, 0)
    yellowish = (255, 255, 140)

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask_yellow = cv2.inRange(img, yellow, yellowish)
    mask_white = cv2.inRange(img, whiteish, white)
    merged_mask = mask_white | mask_yellow
    closing = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_closing)
    dilation = cv2.dilate(closing, kernel_dilation, iterations=1)

    return dilation, closing, merged_mask, mask_white, mask_yellow
