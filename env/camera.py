class Camera:
    def __init__(self, cam_id, client_id, cam_args, record=False):
        self.client_id = client_id
        self.cam_id = cam_id
        self.cam_args = cam_args

        self.window = "Camera_" + str(self.cam_id)

        self.mode = cam_args["mode"]  # or "position"
        self.target = cam_args["target"]

        # For distance mode:
        self.distance = cam_args["distance"]
        self.yaw = cam_args["yaw"]
        self.pitch = cam_args["pitch"]
        self.roll = cam_args["roll"]
        self.up_axis_index = cam_args["up_axis_index"]

        # For position mode:
        self.eye = cam_args["eye"]
        self.up_vec = cam_args["up_vec"]

        # Intrinsics:
        self.width = cam_args["width"]
        self.height = cam_args["height"]
        self.fov = cam_args["fov"]
        self.near = cam_args["near"]
        self.far = cam_args["far"]

        # If camera is already saved somewhere:
        self.view_matrix = cam_args["view_matrix"]
        self.projection_matrix = cam_args["projection_matrix"]

        self.quat = R.from_euler("xyz", [self.yaw, self.pitch, self.roll]).as_quat()
        self.aspect = self.width / self.height

        if (self.view_matrix is None) or (self.view_matrix == "None"):
            if self.mode == "distance":
                self.view_matrix = self.client_id.computeViewMatrixFromYawPitchRoll(
                    self.target,
                    self.distance,
                    self.yaw,
                    self.pitch,
                    self.roll,
                    self.up_axis_index,
                )
            elif self.mode == "position":
                self.view_matrix = self.client_id.computeViewMatrix(
                    self.eye, self.target, self.up_vec
                )

        if (self.projection_matrix is None) or (self.projection_matrix == "None"):
            self.projection_matrix = self.client_id.computeProjectionMatrixFOV(
                self.fov, self.aspect, self.near, self.far
            )

    def capture(self):
        _, _, rgb, depth, segs = self.client_id.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
        )

        # rgb = np.reshape(rgb, (self.width, self.height, 4))
        # rgb = rgb[..., :3]

        # depth = np.reshape(depth, (self.width, self.height))
        rgb = np.reshape(rgb, (self.height, self.width, 4))
        rgb = rgb[..., :3]

        depth = np.reshape(depth, (self.height, self.width))

        return rgb, depth, segs

    def get_pointcloud(self, depth, seg_img=None):
        """Returns a point cloud and its segmentation from the given depth image

        Args:
        -----
            depth (np.array): depth image
            width (int): width of the image
            height (int): height of the image
            view_matrix (np.array): 4x4 view matrix
            proj_matrix (np.array): 4x4 projection matrix
            seg_img (np.array): segmentation image

        Return:
        -------
            pcd (np.array): Nx3 point cloud
            pcd_seg (np.array): N array, segmentation of the point cloud
        """
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(
            np.matmul(proj_matrix, view_matrix)
        )  # Pixel to 3D transformation

        # create a mesh grid with pixel coordinates, by converting 0 to width and 0 to height to -1 to 1
        y, x = np.mgrid[-1 : 1 : 2 / self.height, -1 : 1 : 2 / self.width]
        y *= -1.0  # y is reversed in pixel coordinates

        # Reshape to single dimension arrays
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)

        # Homogenize:
        pixels = np.stack([x, y, z, np.ones_like(z)], axis=1)
        # filter out "infinite" depths:
        fin_depths = np.where(z < 0.99)
        pixels = pixels[fin_depths]

        # Depth z is between 0 to 1, so convert it to -1 to 1.
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        if seg_img is not None:
            seg_img = np.array(seg_img)
            pcd_seg = seg_img.reshape(-1)[fin_depths]  # filter out "infinite" depths
        else:
            pcd_seg = None

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]  # Homogenize in 3D
        points = points[:, :3]  # Remove last axis ones

        return points, pcd_seg, fin_depths

    def move_camera(self):
        self.yaw = max(self.yaw + self.yaw_rate, self.yaw_limit)

        # Update view matrix:
        self.view_matrix = self.client_id.computeViewMatrixFromYawPitchRoll(
            self.target,
            self.distance,
            self.yaw,
            self.pitch,
            self.roll,
            self.up_axis_index,
        )

        # When limit is reached:
        if self.yaw == self.yaw_limit:
            # Exchange yaw limit and initial yaw
            self.yaw_limit, self.initial_yaw = self.initial_yaw, self.yaw_limit
            # Move in the other direction
            self.yaw_rate = -self.yaw_rate

    def record(self):
        rgb, _, _ = self.capture()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        self.out.write(rgb)

        # Move the camera:
        self.move_camera()

    def start_recording(self, plan_folder):
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            os.path.join(plan_folder, self.cam_args["output_file"]),
            self.fourcc,
            self.cam_args["fps"],
            (self.width, self.height),
        )

        self.moving = self.cam_args["moving"]
        self.yaw_limit = self.cam_args["yaw_limit"]
        self.yaw_rate = self.cam_args["yaw_rate"]
        self.initial_yaw = self.yaw

    def stop_recording(self):
        self.out.release()