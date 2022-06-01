from ipywidgets import Box, Layout

import sys
sys.path.append('C:/Users/admin/Downloads/VoxelHashing-master/voxelhashing_offline')
from devices.robot.robot_on_sphere import RobotOnSphere
from calibration.trajectories.trajectory import Trajectory

def make_im_slice(pos, size, step=None):
    return (slice(pos[0] - size // 2, pos[0] + size // 2, step),
            slice(pos[1] - size // 2, pos[1] + size // 2, step))


class CameraCalibrationSphere(Trajectory):
    points = None
    robot = None

    def __init__(self, robot=None):
        if robot is None:
            robot = RobotOnSphere(simulation=True)
        self.robot = robot
        self._stop_streaming = None
        self.points = self.robot.generate_trajectory_points()

    def move_zero(self, velocity):
        self.robot.move_to((.5, .5), velocity)

    def stream_tag(self, realsense, tis_left, tis_right, kinect, phone_left, phone_right):
        realsense_rgb_w = realsense.start_streaming('image', make_im_slice((646, 797), 102))
        tis_left_w = tis_left.start_streaming('image', make_im_slice((1275, 1509), 177))
        tis_right_w = tis_right.start_streaming('image', make_im_slice((1320, 1066), 177))
        kinect_rgb_w = kinect.start_streaming('image', make_im_slice((585, 829), 78))
        phone_left_w = phone_left.start_streaming('image', make_im_slice((2182, 3996), 430, 4))
        phone_right_w = phone_right.start_streaming('image', make_im_slice((2174, 2628), 430, 4))

        def _stop_streaming():
            for camera in [realsense, tis_left, tis_right, kinect, phone_left, phone_right]:
                camera.stop_streaming('image')
        self._stop_streaming = _stop_streaming

        images = [realsense_rgb_w, tis_left_w, phone_left_w, phone_right_w,
                  kinect_rgb_w, tis_right_w]
        for image in images:
            image.width = '220px'
            image.layout.object_fit = 'contain'
        widget = Box(images, layout=Layout(display='flex', flex_flow='row wrap'))
        return widget

    def stop_tag_streaming(self):
        self._stop_streaming()
