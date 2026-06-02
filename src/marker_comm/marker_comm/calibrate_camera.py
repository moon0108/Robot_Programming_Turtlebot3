import argparse
import glob
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_checkerboard(value):
    try:
        cols, rows = value.lower().split('x')
        return int(cols), int(rows)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use COLSxROWS, for example 8x6') from exc


def make_object_points(checkerboard, square_size):
    cols, rows = checkerboard
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def write_ros_camera_yaml(path, camera_name, image_size, camera_matrix, dist_coeffs):
    width, height = image_size
    dist = dist_coeffs.reshape(-1)
    content = f"""image_width: {width}
image_height: {height}
camera_name: {camera_name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{', '.join(f'{v:.12g}' for v in camera_matrix.reshape(-1))}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {len(dist)}
  data: [{', '.join(f'{v:.12g}' for v in dist)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{camera_matrix[0, 0]:.12g}, {camera_matrix[0, 1]:.12g}, {camera_matrix[0, 2]:.12g}, 0.0, {camera_matrix[1, 0]:.12g}, {camera_matrix[1, 1]:.12g}, {camera_matrix[1, 2]:.12g}, 0.0, {camera_matrix[2, 0]:.12g}, {camera_matrix[2, 1]:.12g}, {camera_matrix[2, 2]:.12g}, 0.0]
"""
    path.write_text(content, encoding='utf-8')


def calibrate_camera(
    image_dir,
    checkerboard,
    square_size,
    output_dir,
    camera_name,
    show_corners,
):
    image_paths = sorted(glob.glob(str(image_dir / '*.png')) + glob.glob(str(image_dir / '*.jpg')))
    if not image_paths:
        raise RuntimeError(f'No .png or .jpg images found in {image_dir}')

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    objp = make_object_points(checkerboard, square_size)
    objpoints = []
    imgpoints = []
    image_size = None
    used_images = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f'[skip] could not read {image_path}')
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, checkerboard, flags)

        if not found:
            print(f'[miss] {image_path}')
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined)
        used_images.append(image_path)
        print(f'[ok]   {image_path}')

        if show_corners:
            drawn = cv2.drawChessboardCorners(image, checkerboard, refined, found)
            cv2.imshow('checkerboard_corners', drawn)
            cv2.waitKey(250)

    if show_corners:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(
            f'Only {len(objpoints)} valid checkerboard images found. '
            'Use at least 10, ideally 20-40 images from different angles.'
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / 'camera_calibration.pkl'
    yaml_path = output_dir / 'camera_calibration.yaml'

    calibration_data = {
        'rms_reprojection_error': rms,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'image_size': image_size,
        'checkerboard': checkerboard,
        'square_size': square_size,
        'used_images': used_images,
    }

    with pkl_path.open('wb') as file:
        pickle.dump(calibration_data, file)
    write_ros_camera_yaml(yaml_path, camera_name, image_size, camera_matrix, dist_coeffs)

    print('\nCalibration complete')
    print(f'RMS reprojection error: {rms:.6f}')
    print('Camera matrix:')
    print(camera_matrix)
    print('Distortion coefficients:')
    print(dist_coeffs)
    print(f'Saved pickle: {pkl_path}')
    print(f'Saved ROS YAML: {yaml_path}')


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Calibrate a camera from checkerboard images.')
    parser.add_argument('--image-dir', default='checkerboards', type=Path)
    parser.add_argument('--checkerboard', default='8x6', type=parse_checkerboard)
    parser.add_argument('--square-size', default=0.025, type=float, help='Checker square size in meters.')
    parser.add_argument('--output-dir', default='calibration', type=Path)
    parser.add_argument('--camera-name', default='camera')
    parser.add_argument('--show-corners', action='store_true')
    return parser


def main():
    args = build_arg_parser().parse_args()
    calibrate_camera(
        image_dir=args.image_dir.expanduser(),
        checkerboard=args.checkerboard,
        square_size=args.square_size,
        output_dir=args.output_dir.expanduser(),
        camera_name=args.camera_name,
        show_corners=args.show_corners,
    )


if __name__ == '__main__':
    main()
