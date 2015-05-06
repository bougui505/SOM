import IO
import numpy


class Dihedral:
    def __init__(self, dcdfile, struct):
        """

        :type struct: str
        :type dcdfile: str
        """
        self.dcdfile = dcdfile
        self.struct = struct
        self.traj = IO.Trajectory(dcdfile=self.dcdfile, struct=self.struct)
        self.traj.array = self.traj.array.reshape(self.traj.nframe, self.traj.natom, 3)
        self.backbone = self.traj.array[:, self.backbone_selection]  # atomic coordinates of the backbone

    @property
    def backbone_selection(self):
        """

        :return: a selection of the backbone without O
        """
        atomnames = self.traj.struct.atoms['atomname']
        selection = (atomnames == 'N') | (atomnames == 'CA') | (atomnames == 'C')
        return selection

    def get_phi(self):
        """

        :return: the phi dihedral angle in radian
        """
        atomnames = self.traj.struct.atoms['atomname']
        a1 = atomnames == 'C'  # the notation come from: http://en.wikipedia.org/wiki/Dihedral_angle
        a2 = atomnames == 'N'
        a3 = atomnames == 'CA'
        b1 = a2
        b2 = a3
        b3 = a1
        frame_id = 0
        frame = self.traj.array[frame_id]
        u_a = numpy.cross((frame[a2][1:] - frame[a1][:-1]), (frame[a3][1:] - frame[a1][:-1]))
        u_b = numpy.cross((frame[b2][1:] - frame[b1][1:]), (frame[b3][1:] - frame[b1][1:]))
        #dotp = numpy.dot(u_a, u_b.T).diagonal()
        dotp = numpy.einsum('ij,ji->i', u_a,
                            u_b.T)  # dot product only for the diagonal
                                    # (see: http://stackoverflow.com/a/14759341/1679629)
        norm_u_a = numpy.linalg.norm(u_a, axis=1)
        norm_u_b = numpy.linalg.norm(u_b, axis=1)
        phi = numpy.arccos(dotp / (norm_u_a * norm_u_b))
        return phi
