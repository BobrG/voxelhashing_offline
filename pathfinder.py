import sys 
sys.path.append('C:/Users/admin/Downloads/VoxelHashing-master/voxelhashing_offline')
from calibration.trajectories.camera_sphere import CameraCalibrationSphere

# from skrgbd.data.dataset.params import cam_trajectory, https://github.com/voyleg/dev.sk_robot_rgbd_data
cam_trajectory = CameraCalibrationSphere()
cam_pos_ids = range(100)

light_setups = [
    'flash@best', 'flash@fast', 'ambient@best', 'ambient_low@fast', 'hard_left_bottom_close@best',
    'hard_left_bottom_far@best', 'hard_left_top_close@best', 'hard_left_top_far@best', 'hard_right_bottom_close@best',
    'hard_right_top_close@best', 'hard_right_top_far@best', 'soft_left@best', 'soft_right@best', 'soft_top@best']

kinect_light_setups = [
    'flash', 'ambient', 'hard_left_bottom_close', 'hard_left_bottom_far', 'hard_left_top_close', 'hard_left_top_far',
    'hard_right_bottom_close', 'hard_right_top_close', 'hard_right_top_far', 'soft_left', 'soft_right', 'soft_top']

# from skrgbd.utils import SimpleNamespace, https://github.com/voyleg/dev.sk_robot_rgbd_data
class SimpleNamespace(dict):
    def __init__(self):
        super().__init__()

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        dict.__setitem__(self, key, value)

    def __setitem__(self, key, value):
        object.__setattr__(self, key, value)
        dict.__setitem__(self, key, value)

class ScenePaths(SimpleNamespace):
    def __init__(self, scene_name=None, data_root=None, aux_root=None, raw_scans_root=None):
        self.stl = StlPaths(scene_name, data_root, aux_root, raw_scans_root)
        self.tis_left = TISPaths('tis_left', data_root, scene_name, raw_scans_root)
        self.tis_right = TISPaths('tis_right', data_root, scene_name, raw_scans_root)
        self.kinect_v2 = KinectPaths(data_root, scene_name, raw_scans_root)
        self.phone_left = PhonePaths('phone_left', data_root, scene_name, raw_scans_root)
        self.phone_right = PhonePaths('phone_right', data_root, scene_name, raw_scans_root)
        self.real_sense = RealSensePaths(data_root, scene_name, raw_scans_root)

        self.reprojected = SimpleNamespace()
        self.reprojected.depth = dict()

        for dst_sensor in ['kinect_ir']:
            src_device = 'stl'
            src_variant = 'reconstruction_cleaned'
            for dst_variant in ['raw', 'undistorted']:
                self.reprojected.depth[(src_device, src_variant), (dst_sensor, dst_variant)] = ReprojectedDepthPaths(
                    src_device, src_variant, dst_sensor, dst_variant, data_root, scene_name
                )


class Pathfinder(ScenePaths):
    def __init__(self, data_root=None, aux_root=None, raw_scans_root=None):
        ScenePaths.__init__(self, 'scene_name', data_root, aux_root, raw_scans_root)
        self.data_root = data_root
        self.aux_root = aux_root
        self.raw_scans_root = raw_scans_root

    def __getitem__(self, attr_name):
        if attr_name in {'tis_left', 'tis_right', 'kinect_v2', 'phone_left', 'phone_right', 'real_sense'}:
            cam_name = attr_name
            return getattr(self, cam_name)
        else:
            scene_name = attr_name
            return ScenePaths(scene_name, self.data_root, self.aux_root, self.raw_scans_root)


class StlPaths(SimpleNamespace):
    def __init__(self, scene_name, data_root=None, aux_root=None, raw_scans_root=None):
        self.partial = SimpleNamespace()
        self.partial.raw = f'{raw_scans_root}/{scene_name}/stl/{scene_name}_folder'
        self.partial.aligned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/partial/aligned/{scan_i:04}.ply')
        self.partial.aligned.refined_board_to_world = f'{aux_root}/{scene_name}/stl/partial/refined_board_to_world.pt'
        self.partial.cleaned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/partial/cleaned/{scan_i:04}.ply')

        self.validation = SimpleNamespace()
        self.validation.raw = f'{raw_scans_root}/{scene_name}/stl/{scene_name}_check_folder'
        self.validation.aligned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/validation/aligned/{scan_i:04}.ply')

        self.reconstruction = SimpleNamespace()
        self.reconstruction.pre_cleaned = f'{data_root}/{scene_name}/stl/reconstruction/pre_cleaned.ply'
        self.reconstruction.cleaned = f'{data_root}/{scene_name}/stl/reconstruction/cleaned.ply'

        self.occluded_space = f'{data_root}/{scene_name}/stl/occluded_space.ply'


class ImageSensorPaths(SimpleNamespace):
    def __init__(self, camera_name, modality, data_root=None, scene_name=None, ext='png', light_dependent=True):
        self.calibrated_intrinsics = f'{data_root}/calibration/{camera_name}/{modality}/intrinsics.yaml'
        self.calibrated_extrinsics = f'{data_root}/calibration/{camera_name}/{modality}/images.txt'
        self.pinhole_intrinsics = f'{data_root}/calibration/{camera_name}/{modality}/cameras.txt'
        self.pinhole_pxs_in_raw = f'{data_root}/calibration/{camera_name}/{modality}/pinhole_pxs_in_raw.pt'
        if light_dependent:
            self.undistorted = IndexedPath(
                lambda light_setup_pos_i:
                f'{data_root}/{scene_name}/{camera_name}/{modality}/undistorted/{light_setup_pos_i[0]}/{light_setup_pos_i[1]:04}.{ext}')
        else:
            self.undistorted = IndexedPath(
                lambda pos_i:
                f'{data_root}/{scene_name}/{camera_name}/{modality}/undistorted/{pos_i:04}.{ext}')
        self.refined_extrinsics = f'{data_root}/{scene_name}/{camera_name}/{modality}/images.txt'


class DepthSensorPaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, scene_name=None, light_dependent=False):
        self.calibrated_intrinsics = f'{data_root}/calibration/{camera_name}/ir/intrinsics.yaml'
        self.calibrated_extrinsics = f'{data_root}/calibration/{camera_name}/ir/images.txt'
        self.pinhole_intrinsics = f'{data_root}/calibration/{camera_name}/ir/cameras.txt'
        self.undistortion_model = f'{data_root}/calibration/{camera_name}/depth/undistortion.pt'
        if not light_dependent:
            self.undistorted = IndexedPath(
                lambda pos_i:
                f'{data_root}/{scene_name}/{camera_name}/depth/undistorted/{pos_i:04}.png')
        else:
            self.undistorted = IndexedPath(
                lambda light_setup_pos_i:
                f'{data_root}/{scene_name}/{camera_name}/depth/undistorted/{light_setup_pos_i[0]}/{light_setup_pos_i[1]:04}.png')
        self.refined_extrinsics = f'{data_root}/{scene_name}/{camera_name}/ir/images.txt'


class TISPaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, scene_name=None, raw_scans_root=None):
        self.rgb = ImageSensorPaths(camera_name, 'rgb', data_root, scene_name)
        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')


class KinectPaths(SimpleNamespace):
    def __init__(self, data_root=None, scene_name=None, raw_scans_root=None):
        self.rgb = ImageSensorPaths('kinect_v2', 'rgb', data_root, scene_name)
        self.ir = ImageSensorPaths('kinect_v2', 'ir', data_root, scene_name, light_dependent=False)
        self.depth = DepthSensorPaths('kinect_v2', data_root, scene_name)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')
        self.ir.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{cam_trajectory[pos_i]}_ir.png')
        self.depth.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{cam_trajectory[pos_i]}_depth.png')


class PhonePaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, scene_name=None, raw_scans_root=None):
        self.rgb = ImageSensorPaths(camera_name, 'rgb', data_root, scene_name, ext='jpg')
        self.ir = ImageSensorPaths(camera_name, 'ir', data_root, scene_name, light_dependent=False)
        self.depth = DepthSensorPaths(camera_name, data_root, scene_name)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.jpg')
        self.ir.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{cam_trajectory[pos_i]}_ir.png')
        self.depth.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{cam_trajectory[pos_i]}_depth.png')


class RealSensePaths(SimpleNamespace):
    def __init__(self, data_root=None, scene_name=None, raw_scans_root=None):
        self.rgb = ImageSensorPaths('real_sense', 'rgb', data_root, scene_name)
        self.ir = ImageSensorPaths('real_sense', 'ir', data_root, scene_name)
        self.ir_right = ImageSensorPaths('real_sense', 'ir_right', data_root, scene_name)
        self.depth = DepthSensorPaths('real_sense', data_root, scene_name, light_dependent=True)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')
        self.ir.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_ir.png')
        self.ir_right.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_irr.png')
        self.depth.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{cam_trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_depth.png')


class ReprojectedDepthPaths(SimpleNamespace):
    def __init__(self, src_device, src_variant, dst_sensor, dst_variant, data_root=None, scene_name=None, light_dependent=False):
        self.data_root = data_root
        self.scene_name = scene_name
        self.light_dependent = light_dependent
        self.device_from = f'{src_device}.{src_variant}'
        self.sensor_to = f'{dst_sensor}.{dst_variant}'

    def __getitem__(self, item):
        if not self.light_dependent:
            pos_i = item
            return f'{self.data_root}/{self.scene_name}/reprojected/depth/{self.device_from}@{self.sensor_to}/{pos_i:04}.png`'
        else:
            light_setup, pos_i = item
            return f'{self.data_root}/{self.scene_name}/reprojected/depth/{self.device_from}@{self.sensor_to}/{light_setup}/{pos_i:04}.png`'


class IndexedPath:
    def __init__(self, i_to_path):
        self.i_to_path = i_to_path

    def __getitem__(self, i):
        return self.i_to_path(i)


sensor_to_cam_mode = {
    'real_sense_rgb': ('real_sense', 'rgb'),
    'real_sense_ir': ('real_sense', 'ir'),
    'real_sense_ir_right': ('real_sense', 'ir_right'),
    'kinect_v2_rgb': ('kinect_v2', 'rgb'),
    'kinect_v2_ir': ('kinect_v2', 'ir'),
    'tis_left': ('tis_left', 'rgb'),
    'tis_right': ('tis_right', 'rgb'),
    'phone_left_rgb': ('phone_left', 'rgb'),
    'phone_left_ir': ('phone_left', 'ir'),
    'phone_right_rgb': ('phone_right', 'rgb'),
    'phone_right_ir': ('phone_right', 'ir'),
}
