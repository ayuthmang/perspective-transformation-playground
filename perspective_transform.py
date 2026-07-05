"""
Interactive 4-point perspective transform (planar homography).

Usage:
    python perspective_transform.py path/to/image.png
    python perspective_transform.py path/to/image.png --width 600 --height 400

Click the 4 corners of the region you want to rectify (any order).
Keys while picking:  u = undo last point   r = reset   Esc = cancel
"""

import argparse
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered as [top-left, top-right, bottom-right, bottom-left].

    This ordering must match the destination rectangle, otherwise the warp
    comes out rotated, mirrored, or torn. The trick: the top-left has the
    smallest (x + y), the bottom-right the largest; the top-right has the
    smallest (x - y), the bottom-left the largest.
    """
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # x - y  (note: cv2 points are [x, y])

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]     # top-left
    ordered[2] = pts[np.argmax(s)]     # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def four_point_transform(image, pts, out_size=None):
    """Warp the quadrilateral defined by `pts` to a front-facing rectangle.

    out_size = (W, H)  -> forces a fixed output size.
    out_size = None     -> auto-computes W and H from the edge lengths so the
                           aspect ratio roughly matches what you selected.

    Returns (warped_image, M) where M is the 3x3 homography.
    """
    src = order_points(pts)
    (tl, tr, br, bl) = src

    if out_size is None:
        width_bottom = np.linalg.norm(br - bl)
        width_top = np.linalg.norm(tr - tl)
        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        W = int(round(max(width_bottom, width_top)))
        H = int(round(max(height_right, height_left)))
    else:
        W, H = out_size

    dst = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped, M


def pick_points(image, window="Click 4 corners  (u=undo  r=reset  Esc=cancel)"):
    """Let the user click exactly 4 points, with live visual feedback."""
    pts = []
    clone = image.copy()

    def redraw():
        vis = image.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts)], len(pts) == 4, (0, 255, 0), 2)
        cv2.imshow(window, vis)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                      # Esc
            pts.clear()
            break
        if key in (ord("u"), ord("U")) and pts:
            pts.pop()
            redraw()
        if key in (ord("r"), ord("R")):
            pts.clear()
            redraw()
        if len(pts) == 4:
            cv2.waitKey(400)               # brief pause so the 4th dot shows
            break

    cv2.destroyWindow(window)
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--width", type=int, default=None,
                        help="fixed output width; omit to auto-compute")
    parser.add_argument("--height", type=int, default=None,
                        help="fixed output height; omit to auto-compute")
    parser.add_argument("--out", default="warped.png")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")

    pts = pick_points(image)
    if len(pts) != 4:
        raise SystemExit("Cancelled — need exactly 4 points.")

    out_size = None
    if args.width and args.height:
        out_size = (args.width, args.height)

    warped, M = four_point_transform(image, pts, out_size=out_size)

    print("Homography M =\n", M)
    print("Output size (w, h):", (warped.shape[1], warped.shape[0]))
    cv2.imwrite(args.out, warped)
    print("Saved:", args.out)

    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
