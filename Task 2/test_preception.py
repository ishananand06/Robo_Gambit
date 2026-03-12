"""
Headless test script for perception.py
Runs the pipeline on board images and prints detected board states.
Does not open any GUI windows and provides richer diagnostics.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Monkey-patch cv2.imshow and cv2.waitKey to suppress GUI
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: 0
cv2.destroyAllWindows = lambda *args, **kwargs: None

# Add the task directory to the path dynamically
TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from perception import RoboGambit_Perception

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless regression test for RoboGambit perception")
    parser.add_argument(
        "--input",
        type=Path,
        default=TASK_DIR / "input",
        help="Directory containing board images (default: %(default)s)",
    )
    parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob pattern for images inside --input (default: %(default)s)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Explicit list of image paths to process (overrides --input/--glob)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many images",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print the final board matrices (suppress per-piece diagnostics)",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Skip the synthetic coordinate mapping test",
    )
    return parser.parse_args()


def gather_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.images:
        resolved = []
        for img in args.images:
            candidate = img.expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.exists():
                resolved.append(candidate)
            else:
                print(f"Warning: image {candidate} not found")
        return resolved

    input_dir = args.input.expanduser()
    if not input_dir.is_absolute():
        input_dir = (TASK_DIR / input_dir).resolve()

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return []

    return sorted(input_dir.glob(args.glob))


def test_all_boards(image_paths: list[Path], args: argparse.Namespace):
    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        print("No images found for testing.")
        return

    print("=" * 60)
    print("PERCEPTION PIPELINE TEST")
    print("=" * 60)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"\nFailed to load: {img_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"Image size: {image.shape}")
        print(f"{'='*60}")
        
        perception = RoboGambit_Perception()
        
        # Run the pipeline
        perception.process_image(image)
        
        print(f"\nBoard state:")
        print(perception.board)
        
        if not args.summary:
            non_zero = np.argwhere(perception.board != 0)
            print(f"\nDetected pieces ({len(non_zero)}):")
            for pos in non_zero:
                r, c = pos
                piece_id = perception.board[r][c]
                print(f"  Piece {piece_id} at row={r}, col={c}")
        
        if not args.summary:
            # Print the corner pixels and homography for debugging
            print(f"\nCorner pixels detected: {perception.corner_pixels}")
            if perception.H_matrix is not None:
                print(f"Homography matrix:\n{perception.H_matrix}")
            
                # Verify by transforming corner markers back
                print(f"\nHomography verification (corner markers):")
                for mid, (wx, wy) in perception.corner_world.items():
                    if mid in perception.corner_pixels:
                        px, py = perception.corner_pixels[mid]
                        tw_x, tw_y = perception.pixel_to_world(px, py)
                        err = np.sqrt((tw_x - wx)**2 + (tw_y - wy)**2)
                        print(f"  Marker {mid}: pixel=({px:.1f},{py:.1f}) -> world=({tw_x:.1f},{tw_y:.1f}) expected=({wx},{wy}) error={err:.2f}")

            if perception.last_piece_detections:
                print(f"\nPiece placement diagnostics:")
                for det in perception.last_piece_detections:
                    px, py = det["pixel"]
                    wx, wy = det["world"]
                    row, col = det["cell"]
                    print(
                        f"  Piece {det['piece_id']:>2} | pixel=({px:.1f},{py:.1f}) "
                        f"world=({wx:.1f},{wy:.1f}) -> cell=({row},{col})"
                    )
        
        print()

def test_coordinate_mapping():
    """Test the place_piece_on_board coordinate mapping logic"""
    print("\n" + "="*60)
    print("COORDINATE MAPPING TEST")
    print("="*60)
    
    perception = RoboGambit_Perception()
    
    # Test known world coordinates that should map to specific cells
    test_cases = [
        # (world_x, world_y, expected_row, expected_col, description)
        (250, 250, 0, 0, "Top-left cell center"),
        (150, 250, 0, 1, "Row 0, Col 1"),
        (50, 250, 0, 2, "Row 0, Col 2"),
        (-50, 250, 0, 3, "Row 0, Col 3"),
        (-150, 250, 0, 4, "Row 0, Col 4"),
        (-250, 250, 0, 5, "Row 0, Col 5"),
        (250, 150, 1, 0, "Row 1, Col 0"),
        (250, -250, 5, 0, "Row 5, Col 0"),
        (-250, -250, 5, 5, "Row 5, Col 5"),
        # Edge cases / boundary tests
        (300, 300, 0, 0, "Board origin"),
        (-300, -300, 5, 5, "Board opposite corner (just within -300 boundary)"),
    ]
    
    all_passed = True
    for wx, wy, exp_row, exp_col, desc in test_cases:
        perception.board[:] = 0
        perception.place_piece_on_board(1, wx, wy)
        
        placed = np.argwhere(perception.board != 0)
        if len(placed) == 1:
            r, c = placed[0]
            status = "PASS" if (r == exp_row and c == exp_col) else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  [{status}] {desc}: world=({wx},{wy}) -> ({r},{c}), expected=({exp_row},{exp_col})")
        elif len(placed) == 0:
            print(f"  [FAIL] {desc}: world=({wx},{wy}) -> NOT PLACED, expected=({exp_row},{exp_col})")
            all_passed = False
        else:
            print(f"  [FAIL] {desc}: world=({wx},{wy}) -> MULTIPLE PLACED, expected=({exp_row},{exp_col})")
            all_passed = False
    
    print(f"\nCoordinate mapping: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


def main():
    args = parse_args()
    if not args.skip_mapping:
        test_coordinate_mapping()

    image_paths = gather_image_paths(args)
    test_all_boards(image_paths, args)


if __name__ == "__main__":
    main()