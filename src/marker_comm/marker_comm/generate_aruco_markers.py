import argparse
from pathlib import Path

import cv2


ARUCO_DICTIONARIES = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
}


def parse_ids(value):
    try:
        return [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integer IDs, for example 0,1') from exc


def generate_marker(dictionary, marker_id, pixels, output_dir):
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    else:
        marker_image = cv2.aruco.drawMarker(dictionary, marker_id, pixels)
    output_path = output_dir / f'aruco_{marker_id}.png'
    cv2.imwrite(str(output_path), marker_image)
    return output_path


def try_generate_pdf(image_paths, marker_size_m, output_dir):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    marker_size_mm = marker_size_m * 1000.0
    margin = 20 * mm
    gap = 15 * mm
    page_width, page_height = A4
    x = margin
    y = page_height - margin - marker_size_mm * mm

    pdf_path = output_dir / 'aruco_markers.pdf'
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.setFont('Helvetica', 10)

    for image_path in image_paths:
        if x + marker_size_mm * mm > page_width - margin:
            x = margin
            y -= marker_size_mm * mm + 25 * mm
        if y < margin:
            pdf.showPage()
            pdf.setFont('Helvetica', 10)
            x = margin
            y = page_height - margin - marker_size_mm * mm

        pdf.drawImage(
            str(image_path),
            x,
            y,
            width=marker_size_mm * mm,
            height=marker_size_mm * mm,
            preserveAspectRatio=True,
            mask='auto',
        )
        pdf.drawString(x, y - 6 * mm, f'{image_path.stem} / {marker_size_mm:.1f} mm')
        x += marker_size_mm * mm + gap

    pdf.save()
    return pdf_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Generate OpenCV ArUco marker images.')
    parser.add_argument('--dictionary', default='DICT_4X4_50', choices=sorted(ARUCO_DICTIONARIES))
    parser.add_argument('--ids', default='0,1', type=parse_ids)
    parser.add_argument('--pixels', default=800, type=int)
    parser.add_argument('--marker-size', default=0.05, type=float, help='Printed marker side length in meters.')
    parser.add_argument('--output-dir', default='aruco_markers', type=Path)
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[args.dictionary])
    image_paths = [
        generate_marker(dictionary, marker_id, args.pixels, output_dir)
        for marker_id in args.ids
    ]

    for image_path in image_paths:
        print(f'Saved {image_path}')

    pdf_path = try_generate_pdf(image_paths, args.marker_size, output_dir)
    if pdf_path is None:
        print('PDF not generated because reportlab is not installed.')
        print('PNG files are ready. Print them without page scaling.')
    else:
        print(f'Saved {pdf_path}')
        print('Print the PDF at 100% scale, not fit-to-page.')


if __name__ == '__main__':
    main()
