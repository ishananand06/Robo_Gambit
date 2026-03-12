import math
import cv2
import numpy as np
import sys


class RoboGambit_Perception:

    def __init__(self):
        # PARAMETERS - Camera intrinsics provided by organisers (DO NOT MODIFY)
        self.camera_matrix = np.array([
            [1030.4890823364258, 0, 960],
            [0, 1030.489103794098, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((1, 5))

        # INTERNAL VARIABLES
        self.corner_world = {
            21: (350, 350),
            22: (350, -350),
            23: (-350, -350),
            24: (-350, 350)
        }
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        self.H_matrix = None

        self.board = np.zeros((6, 6), dtype=int)
        self.last_piece_detections = []

        # ARUCO DETECTOR
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,self.aruco_params)

        print("Perception Initialized")


    # DO NOT MODIFY THIS FUNCTION
    def prepare_image(self, image):
        """
        DO NOT MODIFY.
        Performs camera undistortion and grayscale conversion.
        """
        undistorted_image = cv2.undistort(image,self.camera_matrix,self.dist_coeffs,None,self.camera_matrix)
        gray_image = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
        return undistorted_image, gray_image


    # TODO: IMPLEMENT PIXEL → WORLD TRANSFORMATION
    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates into world coordinates using homography.
        Steps:
        1. Ensure homography matrix has been computed.
        2. Format pixel point for cv2.perspectiveTransform().
        3. Return transformed world coordinates.
        """

        if self.H_matrix is None:
            print("Homography matrix not computed yet!")
            return None, None

        # Format pixel as (1, 1, 2) array for perspectiveTransform
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        # Apply homography to get world coordinates
        world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)

        world_x = world_point[0][0][0]
        world_y = world_point[0][0][1]

        return world_x, world_y


    # PARTICIPANTS MODIFY THIS FUNCTION
    def process_image(self, image):
        """
        Main perception pipeline.
        Participants must implement:
        - ArUco detection
        - Homography computation
        - Pixel → world conversion
        - Board reconstruction
        """

        self.board[:] = 0
        self.last_piece_detections = []

        # Preprocess image (Do not modify)
        undistorted_image, gray_image = self.prepare_image(image)

        # Step 1: Detect ArUco markers
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if ids is None:
            print("No markers detected!")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)

        # Step 2: Extract corner marker pixels (IDs 21-24)
        self.corner_pixels = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.corner_world:
                # Compute center pixel of the marker
                marker_corners = corners[i][0]  # shape (4, 2)
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                self.corner_pixels[marker_id] = (center_x, center_y)

        # Step 3: Build pixel and world coordinate matrices
        self.pixel_matrix = []
        self.world_matrix = []
        for marker_id in sorted(self.corner_pixels.keys()):
            px, py = self.corner_pixels[marker_id]
            wx, wy = self.corner_world[marker_id]
            self.pixel_matrix.append([px, py])
            self.world_matrix.append([wx, wy])

        pixel_pts = np.array(self.pixel_matrix, dtype=np.float32)
        world_pts = np.array(self.world_matrix, dtype=np.float32)

        # Step 4: Compute homography matrix
        if len(pixel_pts) >= 4:
            self.H_matrix, status = cv2.findHomography(pixel_pts, world_pts)
        else:
            print(f"Not enough corner markers detected ({len(pixel_pts)}/4)")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        # Step 5: Convert piece markers (IDs 1-10) to world coordinates
        for i, marker_id in enumerate(ids.flatten()):
            if 1 <= marker_id <= 10:
                marker_corners = corners[i][0]  # shape (4, 2)
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])

                world_x, world_y = self.pixel_to_world(center_x, center_y)
                if world_x is not None and world_y is not None:
                    placement = self.place_piece_on_board(marker_id, world_x, world_y)
                    if placement is not None:
                        row, col = placement
                        self.last_piece_detections.append({
                            "piece_id": int(marker_id),
                            "pixel": (float(center_x), float(center_y)),
                            "world": (float(world_x), float(world_y)),
                            "cell": (row, col)
                        })

        # Visualization (Do not modify)
        res = cv2.resize(undistorted_image, (1152,648))
        cv2.imshow("Detected Markers", res)
        self.visualize_board()


    # TODO: IMPLEMENT BOARD PLACEMENT
    def place_piece_on_board(self, piece_id, x_coord, y_coord):

        """
        Places detected piece on the closest board square.

        Board definition:

        6x6 grid
        top-left corner = (300,300)
        square size = 100mm
        """

        board_origin_x = 300
        board_origin_y = 300
        square_size = 100
        board_margin = 20  # tolerate minor pose noise beyond the nominal boundary

        # Reject pieces that fall completely outside of the playable area
        if (abs(x_coord) > board_origin_x + board_margin or
                abs(y_coord) > board_origin_y + board_margin):
            print(f"Piece {piece_id} ignored: world coordinate ({x_coord:.1f}, {y_coord:.1f}) outside board")
            return None

        # Convert world coordinates to grid column and row
        # Board goes from (300,300) at top-left, decreasing by 100mm per cell
        # x_coord goes from 300 (col 0) to -300 (col 5)
        # y_coord goes from 300 (row 0) to -300 (row 5)
        epsilon = 1e-6  # avoid mapping exact boundary values to an out-of-range index
        col_float = (board_origin_x - x_coord) / square_size
        row_float = (board_origin_y - y_coord) / square_size
        col_float = np.clip(col_float, 0.0, 6.0 - epsilon)
        row_float = np.clip(row_float, 0.0, 6.0 - epsilon)

        col = int(math.floor(col_float))
        row = int(math.floor(row_float))

        if not (0 <= row < 6 and 0 <= col < 6):
            print(f"Piece {piece_id} could not be placed: mapped cell ({row},{col}) is invalid")
            return None

        existing_piece = self.board[row][col]
        if existing_piece not in (0, piece_id):
            print(f"Warning: overwriting cell ({row},{col}) containing piece {existing_piece} with {piece_id}")

        self.board[row][col] = piece_id
        return row, col


    # DO NOT MODIFY THIS FUNCTION
    def visualize_board(self):
        """
        Draw a simple 6x6 board with detected piece IDs
        """
        cell_size = 80
        board_img = np.ones((6*cell_size,6*cell_size,3),dtype=np.uint8) * 255

        for r in range(6):
            for c in range(6):
                x1 = c*cell_size
                y1 = r*cell_size
                x2 = x1+cell_size
                y2 = y1+cell_size
                cv2.rectangle(board_img,(x1,y1),(x2,y2),(0,0,0),2)

                piece = int(self.board[r][c])
                if piece != 0:
                    cv2.putText(board_img,str(piece),(x1+25,y1+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Game Board", board_img)


# DO NOT MODIFY
def main():
    # To run code, use python/python3 perception.py path/to/image.png
    if len(sys.argv) < 2:
        print("Usage: python perception.py image.png")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    perception = RoboGambit_Perception()
    perception.process_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()