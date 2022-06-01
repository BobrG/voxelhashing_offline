from abc import ABC, abstractmethod
import datetime
from time import sleep

import numpy as np
from scipy.spatial.transform import Rotation
import socket
from urx.ursecmon import TimeoutException
from tqdm import tqdm
import threading
import torch
import urx

# from skrgbd.utils.logging import logger, tqdm, https://github.com/voyleg/dev.sk_robot_rgbd_data
# from skrgbd.utils.math import make_extrinsics, https://github.com/voyleg/dev.sk_robot_rgbd_data
def make_extrinsics(R, t):
    r"""Make the extrinsics matrix from a rotation matrix and a translation vector.
    
    Parameters
    ----------
    R : torch.Tensor
        of shape [batch_size, 3, 3]
    t : torch.Tensor
        of shape [batch_size, 3]
    
    Returns
    -------
    extrinsics : torch.Tensor
        of shape [batch_size, 4, 4]
    """
    batch_size = len(R)
    extrinsics = R.new_zeros(batch_size, 4, 4)
    extrinsics[:, 3, 3] = 1
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t
    return extrinsics



class Robot(ABC):
    name = 'robot'

    MAX_VELOCITY = .1
    MAX_ACCELERATION = .1
    MAX_JOINT_VELOCITY = (.001, .001, .001, .001, np.deg2rad(20), np.deg2rad(20))

    velocity = .01
    acceleration = .01
    joint_velocity = (.001, .001, .001, .001, .001, .001)

    @property
    @abstractmethod
    def trajectory_name(self):
        ...

    @abstractmethod
    def move_to(self, pos, velocity=.01):
        r"""Moves the robot to the specified position on the trajectory."""
        ...

    @abstractmethod
    def move_home(self, velocity=.01):
        r"""Moves the robot to "home" position on the trajectory, close to the base."""
        ...

    def __init__(self, ip='192.168.0.12', tcp_pos=(0, 0, 0.14, 0, 0, 0), payload=4.5, simulation=False):
        self._simulation = simulation
        if not simulation:
            # logger.debug(f'{self.name}: Start')

            # A lot of code to fully suppress a connection quirk in urx
            orig_excepthook = threading.excepthook

            def tmp_excepthook(args):
                if args.exc_type is socket.timeout:
                    pass
                else:
                    orig_excepthook(args)
            threading.excepthook = tmp_excepthook
            while True:
                try:
                    self.rob = urx.Robot(ip)
                except TimeoutException:
                    sleep(1)
                    continue
                break
            threading.excepthook = orig_excepthook

            self.rob.set_tcp(tcp_pos)
            self.rob.set_payload(payload)
            # logger.debug(f'{self.name}: Start DONE')
        self._tcp_pos = tcp_pos

    def __del__(self):
        if not self._simulation:
            self.stop()
            self.rob.close()

    def stop(self):
        # logger.debug(f'{self.name}: Stop')
        self.rob.stop()
        # logger.debug(f'{self.name}: Stop DONE')

    def _set_tcp(self):
        self.rob.set_tcp(self._tcp_pos)

    def set_velocity(self, velocity):
        r"""Set TCP movement velocity and acceleration.

        Parameters
        ----------
        velocity : float
            Velocity, m/s. Acceleration is set to velocity / 1 s.
        """
        self.velocity = velocity
        self.acceleration = velocity
        self._safety_check()

    def rest(self, velocity=.01):
        r"""Moves the robot to "rest" position."""
        if np.any(np.array(self._tcp_pos) != np.array((0, 0, 0.14, 0, 0, 0))):
            raise RuntimeError('Resting with a non-default TCP pos is dangerous')
        self.move_home(velocity)
        self.lookat([0.39, -0.16, 0.32], [10., -0.16, 0.32], [0, 0, 1.])

    def lookat(self, pos, lookat, up):
        r"""Move TCP to `pos` and look to `lookat`.

        Parameters
        ----------
        pos : array-like
            Position in Base coordinate system, (x, y, z).
        lookat : array-like
            Point of view in Base coordinate system, (x, y, z).
        up : array-like
            The "up" direction, (x, y, z).

        Returns
        -------
        pose : math3d.Transform
            Final transform from Base to TCP.
        """
        self._set_tcp()
        self._safety_check()
        pos = torch.as_tensor(pos)
        lookat = torch.as_tensor(lookat)
        up = torch.as_tensor(up)
        rxryrz = calculate_tcp_rotation(pos, lookat, up).as_rotvec()
        xyz = pos.numpy()
        return self.rob.movel([*xyz, *rxryrz], acc=self.acceleration, vel=self.velocity)

    def move_over_points(self, points, velocity=0.01,
                         closure=None, closure_args=None, closure_kwargs=None, show_progress=True):
        r"""Move robot over a set of points on the trajectory surface, call closure at each point.

        Parameters
        ----------
        points : iterable of iterable of float
            [(x1, y1, ...), (x2, y2, ...), ...]
        velocity : float
            Endpoint movement velocity.
        closure : callable
            If not None, call this at each point, as closure(point_id, coords, *closure_args, **closure_kwargs).
        closure_args : iterable
        closure_kwargs : dict
        """
        if show_progress:
            points = tqdm(points, eta_format=True)
        for coords in points:
            self.move_to(coords, velocity=velocity)
            if closure is not None:
                if closure_args is None:
                    closure_args = tuple()
                if closure_kwargs is None:
                    closure_kwargs = dict()
                closure(self.get_point_id(*coords), coords, *closure_args, **closure_kwargs)

    def get_point_id(self, *coords):
        return f'{self.trajectory_name}@' + ','.join(f'{c:.3}' for c in coords)

    def get_robot_to_tcp_extrinsics(self):
        R = torch.from_numpy(np.linalg.inv(self.rob.get_pose().orient.get_matrix()))
        p = torch.from_numpy(self.rob.get_pose().get_pos().array)
        t = -R @ p
        return make_extrinsics(R.unsqueeze(0), t.unsqueeze(0)).squeeze(0).numpy()

    def _safety_check(self):
        if self.velocity > Robot.MAX_VELOCITY or self.acceleration > Robot.MAX_ACCELERATION:
            raise ValueError(f'Velocity is {self.velocity}, max is {Robot.MAX_VELOCITY}; acceleration is {self.acceleration}, max is {Robot.MAX_ACCELERATION}')


def calculate_tcp_rotation(pos, lookat, up):
    r"""Calculates rotation of TCP from its position, point of view and the up direction.
    After the rotation, TCP and the point of view lie on the Z axis of TCP,
    and the X axis of TCP is aligned with cross(up, new_z).

    Parameters
    ----------
    pos : torch.Tensor
        of shape [**, 3], in meters.
    lookat : torch.Tensor
        of shape [**, 3], in meters.
    up : torch.Tensor
        of shape [**, 3]

    Returns
    -------
    rotation : Rotation
    """
    min_vector_norm = .1  # 10 cm

    z_in_base = lookat - pos
    z_norm = z_in_base.norm(dim=-1, keepdim=True)
    if (z_norm < min_vector_norm).any():
        raise ValueError(f'lookat is {z_norm} meters away from pos.'
                         ' This is too close and may result in a nonstable movement')
    z_in_base /= z_norm

    x_in_base = up.cross(z_in_base, dim=-1)
    x_norm = x_in_base.norm(dim=-1, keepdim=True)
    if (x_norm < min_vector_norm).any():
        raise ValueError('Up direction is too close to the lookat direction,'
                         ' which may result in a nonstable movement')
    x_in_base /= x_norm

    y_in_base = z_in_base.cross(x_in_base, dim=-1)
    y_in_base /= y_in_base.norm(dim=-1, keepdim=True)

    base_to_tcp_rot_matrix = torch.stack([x_in_base, y_in_base, z_in_base], dim=-1)
    return Rotation.from_matrix(base_to_tcp_rot_matrix.numpy())

