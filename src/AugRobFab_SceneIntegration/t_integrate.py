import os
import glob

import numpy as np
import open3d as o3d
import open3d.core as o3c

from AugRobFab_SceneIntegration.util import import_intrinsic_calib

class TSDF_Integration():
	"""
	A class for integrating TSDF volumes from extrinsic poses, depth, and color (optional) images.
	"""

	def __init__(
			self,
			dir,
			intrinsic_matrix="intrinsic.json",
			voxel_size=3.0/512,
			block_count=100000,
			block_resolution=8,
			depth_max=1.0,
			depth_scale=1000.0,
			device='CUDA:0',
			depth_folder_name = 'depth',
			color_folder_name = None,
			integrate_color=False
		) -> None:
		"""
		Initializes the TSDF_Integration instance.

		Args:
			dir: Directory containing the data for TSDF integration.
			intrinsic_matrix: Path to the JSON file containing intrinsic camera matrix.
			voxel_size: Size of each voxel in the TSDF volume.
			block_count: Number of blocks in the TSDF volume.
			block_resolution: Resolution of each block in the TSDF volume.
			depth_max: Maximum depth value for integration.
			depth_scale: Scale factor for depth values.
			device: Computation device to be used.
			depth_folder_name: Folder name containing depth images.
			color_folder_name: Folder name containing color images.
			integrate_color: Flag to indicate whether to integrate color information.
		"""
		
		# Initialization of the TSDF volume and other parameters
		self.device = o3d.core.Device(device)

		self.dir = dir
		
		self.intrinsic_matrix = self._tensorize_intrinsic_matrix(
			import_intrinsic_calib(self.dir, fn=intrinsic_matrix)[0]
			)

		self.voxel_size = voxel_size
		self.block_count = block_count
		self.block_resolution = block_resolution

		self.depth_max = depth_max
		self.depth_scale = depth_scale

		self.integrate_color = color_folder_name is not None
		self.color_folder_name = color_folder_name
		self.depth_folder_name = depth_folder_name


		self.extrinsics = []

		if self.integrate_color:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight', 'color'),
				attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
				attr_channels=((1), (1), (3)),
				voxel_size=self.voxel_size,
				block_resolution=self.block_resolution,
				block_count=self.block_count,
				device=self.device)
		else:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight'),
				attr_dtypes=(o3c.float32, o3c.float32),
				attr_channels=((1), (1)),
				voxel_size=self.voxel_size,
				block_resolution=self.block_resolution,
				block_count=self.block_count,
				device=self.device)

		self.mesh = o3d.geometry.TriangleMesh()
		self.extrinsic_lineset = o3d.geometry.LineSet()

	def integrate_frame(self, extrinsic_pose, depth_image, color_image=None):
		"""
		Integrates a single frame into the TSDF volume.

		Args:
			extrinsic_pose: The pose of the camera for the current frame.
			depth_image: Depth image to be integrated.
			color_image: Optional color image to be integrated.
		"""

		t_extrinsic = self._tensorize_extrinsic_pose(extrinsic_pose)
		self.extrinsics.append(t_extrinsic)
		t_depth = self._tensorize_depth_image(depth_image)

		frustum_block_coords = self.vbg.compute_unique_block_coordinates(
			t_depth,
			self.intrinsic_matrix,
			t_extrinsic,
			self.depth_scale,
			self.depth_max).to(self.device)

		if self.integrate_color and color_image is not None:
			color = self._tensorize_color_image(color_image)
			self.vbg.integrate(
				frustum_block_coords,
				t_depth,
				color,
				self.intrinsic_matrix,
				t_extrinsic,
				self.depth_scale,
				self.depth_max)
		else:
			self.vbg.integrate(
				frustum_block_coords,
				t_depth,
				self.intrinsic_matrix,
				t_extrinsic,
				self.depth_scale,
				self.depth_max)
			
	def integrate_batch(self):
		"""
		Integrates a batch of frames into the TSDF volume.
		"""

		if self.integrate_color:
			depth_paths, color_paths = self._load_rgbd_file_names()
		else:
			depth_paths = self._load_depth_file_names()

		trajectory_path = os.path.join(self.dir, "trajectory.log")
		extrinsic_poses = self._load_extrinsics(trajectory_path)

		for i in range(len(depth_paths)):
			depth = o3d.t.io.read_image(depth_paths[i]).to(self.device)
			if self.integrate_color:
				color = o3d.t.io.read_image(color_paths[i]).to(self.device)
			else:
				color = None
			print(f'Integrating {i+1}/{len(depth_paths)}...', end='')
			self.integrate_frame(
				extrinsic_pose=extrinsic_poses[i],
				depth_image=depth,
				color_image=color
				)
			print('Done!')

	def _tensorize_depth_image(self, depth_image):
		"""
		Converts a depth image to a tensor format.

		Args:
			depth_image: Depth image to be converted.

		Returns:
			Tensor representation of the depth image.
		"""

		if isinstance(depth_image, o3d.cuda.pybind.t.geometry.Image):
			return depth_image.to(self.device)
		return o3d.t.geometry.Image(o3c.Tensor(depth_image)).to(self.device)

	def _tensorize_color_image(self, color_image):
		"""
		Converts a color image to a tensor format.

		Args:
			color_image: Color image to be converted.

		Returns:
			Tensor representation of the color image.
		"""

		if isinstance(color_image, o3d.cuda.pybind.t.geometry.Image):
			return color_image.to(self.device)
		return o3d.t.geometry.Image(o3c.Tensor(color_image)).to(self.device)

	def _tensorize_intrinsic_matrix(self, intrinsic_matrix):
		"""
		Converts an intrinsic matrix to a tensor format.

		Args:
			intrinsic_matrix: Intrinsic matrix to be converted.

		Returns:
			Tensor representation of the intrinsic matrix.
		"""

		assert (intrinsic_matrix.shape == (3, 3))

		return o3c.Tensor(intrinsic_matrix, o3c.Dtype.Float64)

	def _tensorize_extrinsic_pose(self, extrinsic_pose):
		"""
		Converts an extrinsic pose to a tensor format.

		Args:
			extrinsic_pose: Extrinsic pose to be converted.

		Returns:
			Tensor representation of the extrinsic pose.
		"""

		if isinstance(extrinsic_pose, o3d.cuda.pybind.core.Tensor):
			return extrinsic_pose

		assert (extrinsic_pose.shape == (4, 4))

		return o3c.Tensor(extrinsic_pose, o3c.Dtype.Float64)

	def _load_depth_file_names(self):
		"""
		Loads and returns the names of depth image files.

		Returns:
			List of depth image file names.
		"""

		depth_folder = os.path.join(self.dir, self.depth_folder_name)

		# Only 16-bit png depth is supported
		depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
		n_depth = len(depth_file_names)
		if n_depth == 0:
			print('Depth image not found in {}, abort!'.format(depth_folder))
			return []

		return sorted(depth_file_names)

	def _load_rgbd_file_names(self):
		"""
		Loads and returns the names of depth and color image files.

		Returns:
			Tuple containing lists of depth and color image file names.
		"""

		depth_file_names = self._load_depth_file_names()
		if len(depth_file_names) == 0:
			return [], []

		color_folder = os.path.join(self.dir, self.color_folder_name)
		extensions = ['*.png', '*.jpg']
		for ext in extensions:
			color_file_names = glob.glob(os.path.join(color_folder, ext))
			if len(color_file_names) == len(depth_file_names):
				return depth_file_names, sorted(color_file_names)

		depth_folder = os.path.join(self.dir, self.depth_folder_name)
		print('Found {} depth images in {}, but cannot find matched number of '
			'color images in {} with extensions {}, abort!'.format(
				len(depth_file_names), depth_folder, color_folder, extensions))
		return [], []
	
	def _load_extrinsics(self, path_trajectory):
		"""
		Loads extrinsic camera poses from a given file.

		Args:
			path_trajectory: Path to the file containing the camera trajectory. This can be a '.log' or '.json' file.

		Returns:
			A list of extrinsic poses in tensor format.
		"""

		extrinsics = []
		# For either a fragment or a scene
		if path_trajectory.endswith('log'):
			data = o3d.io.read_pinhole_camera_trajectory(path_trajectory)
			for param in data.parameters:
				extrinsics.append(param.extrinsic)

		# Only for a fragment
		elif path_trajectory.endswith('json'):
			data = o3d.io.read_pose_graph(path_trajectory)
			for node in data.nodes:
				extrinsics.append(np.linalg.inv(node.pose))

		return list(map(lambda x: o3d.core.Tensor(x, o3d.core.Dtype.Float64),
					extrinsics))

	def extract_trianglemesh(self, file_name='mesh.ply'):
		"""
		Extracts a triangle mesh from the TSDF volume.

		Args:
			file_name: Name of the file to save the mesh. Default is 'mesh.ply'.

		Returns:
			Extracted triangle mesh.
		"""

		mesh = self.vbg.extract_triangle_mesh()
		mesh = mesh.to_legacy()

		if file_name is not None:
			path = os.path.join(self.dir, file_name)
			o3d.io.write_triangle_mesh(file_name, mesh)
		return mesh

	def extract_pointcloud(self, file_name='pcd.ply'):
		"""
		Extracts a point cloud from the TSDF volume.

		Args:
			file_name: Name of the file to save the point cloud. Default is 'pcd.ply'.

		Returns:
			Extracted point cloud.
		"""

		pcd = self.vbg.extract_point_cloud()
		pcd = pcd.to_legacy()

		if file_name is not None:
			path = os.path.join(self.dir, file_name)
			o3d.io.write_point_cloud(file_name, pcd)
		return pcd

	def visualize(self):
		"""
		Visualizes the integrated TSDF volume and camera poses.
		"""

		if not self.visualize:
			raise Exception(
				"Visualizer has not been initiated. Please use visualize = True when creating a TSDF_Integration instance.")

		self.mesh = self.extract_trianglemesh()
		self.extrinsic_lineset = self.lineset_from_extrinsics()

		o3d.visualization.draw_geometries([self.mesh, self.extrinsic_lineset])

	def lineset_from_extrinsics(self):
		"""
		Creates a lineset from extrinsic poses for visualization.

		Returns:
			Generated lineset.
		"""
		
		POINTS_PER_FRUSTUM = 5
		EDGES_PER_FRUSTUM = 8

		points = []
		colors = []
		lines = []

		cnt = 0
		for extrinsic in self.extrinsics:
			pose = np.linalg.inv(extrinsic.numpy())

			l = 0.01
			points.append((pose @ np.array([0, 0, 0, 1]).T)[:3])
			points.append((pose @ np.array([2*l, l, 2 * l, 1]).T)[:3])
			points.append((pose @ np.array([2*l, -l, 2 * l, 1]).T)[:3])
			points.append((pose @ np.array([-2*l, -l, 2 * l, 1]).T)[:3])
			points.append((pose @ np.array([-2*l, l, 2 * l, 1]).T)[:3])

			lines.append([cnt + 0, cnt + 1])
			lines.append([cnt + 0, cnt + 2])
			lines.append([cnt + 0, cnt + 3])
			lines.append([cnt + 0, cnt + 4])
			lines.append([cnt + 1, cnt + 2])
			lines.append([cnt + 2, cnt + 3])
			lines.append([cnt + 3, cnt + 4])
			lines.append([cnt + 4, cnt + 1])

			for i in range(0, EDGES_PER_FRUSTUM):
				colors.append(np.array([1, 0, 0]))

			cnt += POINTS_PER_FRUSTUM

		for i in range(len(self.extrinsics) - 1):
			s = i
			t = i + 1
			lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
			colors.append(np.array([0, 1, 0]))

		lineset = o3d.geometry.LineSet()
		lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
		lineset.lines = o3d.utility.Vector2iVector(
			np.vstack(lines).astype(int))
		lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

		return lineset
