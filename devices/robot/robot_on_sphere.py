import numpy as np
import torch

import sys
sys.path.append('C:/Users/admin/Downloads/VoxelHashing-master/voxelhashing_offline')
from devices.robot.robot import Robot

# from skrgbd.utils.math import spherical_to_cartesian, sphere_arange_theta, sphere_arange_phi, https://github.com/voyleg/dev.sk_robot_rgbd_data
def spherical_to_cartesian(rho_theta_phi):
    r"""Converts spherical coordinates to cartesian.

    Parameters
    ----------
    rho_theta_phi : torch.Tensor
        of shape [**, 3].

    Returns
    -------
    xyz : torch.Tensor
        of shape [**, 3]
    """
    rho = rho_theta_phi[..., 0]
    theta = rho_theta_phi[..., 1]
    phi = rho_theta_phi[..., 2]
    x = rho * theta.cos() * phi.sin()
    y = rho * theta.sin() * phi.sin()
    z = rho * phi.cos()
    return torch.stack([x, y, z], -1)

def sphere_arange_theta(r, phi, theta_min, theta_max, step, flip=False, eps=1e-7):
    r"""For a given polar angle phi and a sphere radius r generates the values of the azimuth in the range
    [theta_min, theta_max] so that the respective points are placed on the sphere with the given step.

    Parameters
    ----------
    r : float
    phi : float
    theta_min : float
    theta_max : float
    step : float
    flip : bool
        If True, start the values from theta_max.
    eps : float
        Small value added to the right end of the range to include the end point.

    Returns
    -------
    theta : torch.Tensor
        of shape [values_n]
    """
    theta_step = np.arccos(1 - .5 * (step / (r * np.sin(phi))) ** 2)
    if not flip:
        theta = torch.arange(theta_min, theta_max + eps, theta_step)
    else:
        theta = torch.arange(theta_max, theta_min - eps, -theta_step)
    return theta

def sphere_arange_phi(r, phi_min, phi_max, step, flip=False, eps=1e-7):
    r"""For a given sphere radius r generates the values of the polar angle in the range [phi_min, phi_max] so that
    the respective points are placed on the sphere with the given step.

    Parameters
    ----------
    phi_min : float
    phi_max : float
    step : float
    flip : bool
        If True, start the values from phi_max.
    eps : float
        Small value added to the right end of the range to include the end point.

    Returns
    -------
    phi : torch.Tensor
        of shape [values_n]
    """
    phi_step = np.arccos(1 - .5 * (step / r) ** 2)
    if not flip:
        phi = torch.arange(phi_min, phi_max + eps, phi_step)
    else:
        phi = torch.arange(phi_max, phi_min - eps, -phi_step)
    return phi    


class _RobotOnSphere(Robot):
    r"""Robot moving on a theta-phi rectangle on a sphere around the object.

    Parameters
    ----------
    radius : {'x_small', 'small', 'medium', 'large'}
        40 cm, 50 cm, 60 cm, and 70 cm
    """

    theta_min = None
    theta_max = None
    phi_min = None
    phi_max = None
    sphere_center = None
    sphere_radius = None
    up = None

    _tcp_y = None

    def __init__(self, radius='large', simulation=False):
        Robot.__init__(self, tcp_pos=(0, self._tcp_y, .14, 0, 0, 0), simulation=simulation)
        self._set_parameters(radius=radius)

    def _set_parameters(self, radius):
        r"""Do not change these parameters without careful testing with the robot."""
        center_z = .420 - self._tcp_y

        table_length = 1.50
        from_edge_to_object_x = .15
        from_other_edge_to_base_x = .25 / 2 + .05
        center_x = table_length - from_edge_to_object_x - from_other_edge_to_base_x

        from_edge_to_object_y = .35
        from_edge_to_base_y = .80
        center_y = from_edge_to_object_y - from_edge_to_base_y

        self.sphere_center = torch.tensor([center_x, center_y, center_z])

        if radius == 'large':
            max_object_size = .30
            min_distance_to_object = .55
        elif radius == 'medium':
            max_object_size = .20
            min_distance_to_object = .50
        elif radius == 'small':
            max_object_size = .10
            min_distance_to_object = .45
        elif radius == 'x_small':
            max_object_size = .10
            min_distance_to_object = .35
        else:
            raise ValueError(f'Invalid radius value {radius}')
        self.sphere_radius = min_distance_to_object + max_object_size / 2
        self.max_radius = .55 + .30 / 2
        self.min_radius = .35 + .10 / 2

        self.theta_min = np.deg2rad(125)
        self.theta_max = np.deg2rad(180)
        self.phi_min = np.deg2rad(45)
        self.phi_max = np.deg2rad(85)

        self.up = torch.tensor([0, 0, 1.])

    def move_to(self, pos, velocity=0.01):
        r"""Moves robot to the position pos.
        
        Parameters
        ----------
        pos : iterable of float
            (theta, phi) or (theta, phi, r), 0 <= theta <= 1, 0 <= phi <= 1,
            where (0, 0) corresponds to the bottom left corner, and (1, 1) corresponds to the top right corner.
        velocity : float
            Velocity of movement, 0.1 at max.
        """
        if len(pos) == 2:
            theta, phi = pos
            r = self.sphere_radius
        else:
            theta, phi, r = pos
        if (theta < 0.0 or theta > 1.0) or (phi < 0.0 or phi > 1.0) or (r < self.min_radius or r > self.max_radius):
            raise ValueError
        theta = self.theta_min + (self.theta_max - self.theta_min) * theta
        phi = self.phi_min + (self.phi_max - self.phi_min) * (1 - phi)
        pos = self.sphere_center + spherical_to_cartesian(torch.tensor([r, theta, phi]))

        self.set_velocity(velocity)
        return self.lookat(pos, self.sphere_center, self.up)

    def move_home(self, velocity=.01):
        r"""Moves the robot to "home" position on the trajectory, close to the base."""
        return self.move_to((.7, 0), velocity)

    def generate_trajectory_points(self, step=.0561, add_random=None):
        r"""Generates a trajectory with evenly spaced points.
        The trajectory starts in (1, 0), then goes all the way to the left, then one step up,
        then all the way to the right, then one step up, and so on.

        Parameters
        ----------
        step : float
            Distance between points in meters.
            The default value generates 100 points in 9 rows.
        add_random : iterable of float or None
            (theta_sigma, phi_sigma, r_sigma) if not None, for each point add a randomized version.

        Returns
        -------
        points : torch.Tensor
            of shape [points_n, 2] or [points_n, 3] if add_random is not None.
        """
        phis = []
        thetas = []

        # Calculate the points in spherical coordinates
        flip = True
        for phi in sphere_arange_phi(self.sphere_radius, self.phi_min, self.phi_max, step, flip=True):
            theta = sphere_arange_theta(self.sphere_radius, phi, self.theta_min, self.theta_max, step, flip)
            flip = not flip
            thetas.append(theta)
            phis.append(torch.full_like(theta, phi))
        theta = torch.cat(thetas)
        phi = torch.cat(phis)

        if add_random is not None:
            theta_sigma, phi_sigma, r_sigma = add_random
            r = torch.full_like(theta, self.sphere_radius)

            random_theta = theta + torch.randn_like(theta).clamp_(-3, 3).mul_(theta_sigma)
            random_phi = phi + torch.randn_like(phi).clamp_(-3, 3).mul_(phi_sigma)
            random_r = r + torch.randn_like(r).clamp_(-3, 3).mul_(r_sigma)

            theta = torch.stack([theta, random_theta], -1).ravel()
            phi = torch.stack([phi, random_phi], -1).ravel()
            r = torch.stack([r, random_r], -1).ravel()

        # Normalize the coordinates
        theta = ((theta - self.theta_min) / (self.theta_max - self.theta_min)).clamp_(0, 1)
        phi = (1 - (phi - self.phi_min) / (self.phi_max - self.phi_min)).clamp_(0, 1)

        if add_random is not None:
            r = r.clamp_(self.min_radius, self.max_radius)
            points = torch.stack([theta, phi, r], 1)
        else:
            points = torch.stack([theta, phi], 1)
        return points


class RobotOnSTLSphere(_RobotOnSphere):
    # The elevation of the object above the table is chosen so that the object is centered on STL cameras
    # with the lowest possible sphere center w.r.t the table.
    # The center of the object is ~ 35 cm above the table.
    trajectory_name = 'stl_sphere'
    _tcp_y = .115

    def rest(self, velocity=.01):
        r"""Moves the robot to "rest" position."""
        if np.any(np.array(self._tcp_pos) != np.array((0, .115, 0.14, 0, 0, 0))):
            raise RuntimeError('Resting with a non-default TCP pos is dangerous')
        self.move_home(velocity)
        self.lookat([0.39, -0.16, 0.435], [10., -0.16, 0.435], [0, 0, 1.])

    def generate_trajectory_points(self, step=.12):
        return super().generate_trajectory_points(step=step)


class RobotOnSphere(_RobotOnSphere):
    # The elevation of the sphere above the table is chosen so that the object with the same elevation above the table
    # as in for RobotOnSTLSphere is as centered as possible on all the other cameras. The object then is slightly in the
    # top on the phones and slightly in the bottom on all the other cameras.
    trajectory_name = 'sphere'
    _tcp_y = 0
