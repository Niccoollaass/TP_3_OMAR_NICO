import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="SIFT", help="Detector: GFTT ORB SIFT")
    parser.add_argument("-n", "--nbKp", default=None, type=int, help="Number of keypoints")
    parser.add_argument("-d", "--descriptor", default=True, type=bool, help="Compute descriptors")
    parser.add_argument("-m", "--matching", default="NORM_L2", help="Matching norm")
    parser.add_argument("-i1", "--image1", default="./image0.jpeg", help="Image 1 (leftmost)")
    parser.add_argument("-i2", "--image2", default="./image1.jpeg", help="Image 2")
    parser.add_argument("-i3", "--image3", default="./image2.jpeg", help="Image 3")
    parser.add_argument("-i4", "--image4", default=None, help="Image 4 (rightmost)")
    return parser


def test_load_image(img):
    if img is None or img.size == 0:
        print("Could not load image")
        exit(1)


def load_color_image(path):
    if path is None:
        return None
    img = cv.imread(path)
    test_load_image(img)
    return img


def display_image(img, name):
    cv.namedWindow(name)
    cv.imshow(name, img)


def feature_detector(det_type, gray, nb):
    if gray is None:
        return None

    match det_type.upper():
        case "GFTT":
            print("GFTT not implemented yet")
            sys.exit(1)
        case "ORB":
            orb = cv.ORB_create(nfeatures=nb if nb is not None else 500)
            return orb.detect(gray, None)
        case _:
            sift = cv.SIFT_create(nb if nb is not None else 0)
            return sift.detect(gray, None)


def feature_extractor(det_type, gray, kp):
    if gray is None or kp is None:
        return None

    det_type = det_type.upper()

    if det_type == "GFTT":
        return None
    if det_type == "ORB":
        extractor = cv.ORB_create()
    else:
        extractor = cv.SIFT_create()

    _, desc = extractor.compute(gray, kp)
    return desc


def get_norm(norm_name):
    norm_name = norm_name.upper()
    if norm_name == "NORM_L1":
        return cv.NORM_L1
    if norm_name == "NORM_L2":
        return cv.NORM_L2
    if norm_name == "NORM_HAMMING":
        return cv.NORM_HAMMING
    if norm_name == "NORM_HAMMING2":
        return cv.NORM_HAMMING2
    return cv.NORM_L2


def match_descriptors(matching_type, desc1, desc2, a=3.0):
    if desc1 is None or desc2 is None:
        return []

    norm = get_norm(matching_type)
    bf = cv.BFMatcher(normType=norm, crossCheck=True)

    matches = bf.match(desc1, desc2)
    if len(matches) == 0:
        return []

    minDist = min(m.distance for m in matches)
    seuil = a * minDist

    return [m for m in matches if m.distance <= seuil]


def stitch_two_images(img_left, img_right, det_type, nbKp, matching_type, a=3.0):
    if img_left is None or img_right is None:
        return None

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    kp_left = feature_detector(det_type, gray_left, nbKp)
    kp_right = feature_detector(det_type, gray_right, nbKp)

    desc_left = feature_extractor(det_type, gray_left, kp_left)
    desc_right = feature_extractor(det_type, gray_right, kp_right)

    matches = match_descriptors(matching_type, desc_left, desc_right, a)

    if len(matches) < 4:
        print("Not enough matches to compute homography")
        return None

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv.findHomography(pts_right, pts_left, cv.RANSAC)
    if H is None:
        print("Homography failed")
        return None

    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    pano_width = w_left + w_right
    pano_height = max(h_left, h_right)

    warped = cv.warpPerspective(img_right, H, (pano_width, pano_height))
    warped[0:h_left, 0:w_left] = img_left

    return warped


def stitch_sequence(images, det_type, nbKp, matching_type, a=3.0):
    pano = images[0]

    for i in range(1, len(images)):
        if images[i] is None:
            continue
        print(f"Stitching image {i+1}...")
        pano = stitch_two_images(pano, images[i], det_type, nbKp, matching_type, a)
        if pano is None:
            print("Failed during stitching")
            return None

    return pano


def main():
    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    img1 = load_color_image(args["image1"])
    img2 = load_color_image(args["image2"])
    img3 = load_color_image(args["image3"])
    img4 = load_color_image(args["image4"])

    images = [img1, img2, img3, img4]

    panorama = stitch_sequence(
        images,
        args["kp"],
        args["nbKp"],
        args["matching"],
        a=3.0
    )

    if panorama is not None:
        display_image(panorama, "Panorama final")

    key = 0
    while key != ESC_KEY and key != Q_KEY:
        key = cv.waitKey(1)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
