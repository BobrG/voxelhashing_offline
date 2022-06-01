from abc import ABC, abstractmethod
import re


class Trajectory(ABC):
    point_dict = None

    @abstractmethod
    def points(self):
        r"""
        Returns
        -------
        points : torch.Tensor
            of shape [points_n, 2].
        """
        ...

    @abstractmethod
    def robot(self): ...

    @property
    def name(self):
        return self.robot.trajectory_name

    def __getitem__(self, i):
        if self.point_dict is None:
            self._init_point_dict()
        return self.point_dict[i]

    def __contains__(self, i):
        if self.point_dict is None:
            self._init_point_dict()
        return i in self.point_dict

    def __len__(self):
        return len(self.points)

    def point_id_from_filename(self, filename):
        match = re.match(f'^{self.name}@[\d\.]*(,[\d\.]*)*', filename)  # matches {self.name}@float[,float,float,...]
        if match is None:
            return None
        return match.group()

    def _init_point_dict(self):
        self.point_dict = dict()
        for i, point in enumerate(self.points):
            point_id = self.robot.get_point_id(*point)
            self.point_dict[i] = point_id
            self.point_dict[point_id] = i

