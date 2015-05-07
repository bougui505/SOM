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

    def get_phi(self, frame_id):
        """

        :param frame_id: the id of the frame to compute the psi angle on
        :type frame_id: int
        :return: the phi dihedral angle in radian for the frame given by frame_id
        """
        atomnames = self.traj.struct.atoms['atomname']
        a1 = atomnames == 'C'  # the notation come from: http://en.wikipedia.org/wiki/Dihedral_angle
        a2 = atomnames == 'N'
        a3 = atomnames == 'CA'
        frame = self.traj.array[frame_id]
        b1 = a2
        b2 = a3
        b3 = a1
        u_a = numpy.cross((frame[a2][1:] - frame[a1][:-1]), (frame[a3][1:] - frame[a1][:-1]))
        u_b = numpy.cross((frame[b2][1:] - frame[b1][1:]), (frame[b3][1:] - frame[b1][1:]))
        # below is the dot product only for the diagonal (see: http://stackoverflow.com/a/14759341/1679629)
        dotp = numpy.einsum('ij,ji->i', u_a, u_b.T)
        norm_u_a = numpy.linalg.norm(u_a, axis=1)
        norm_u_b = numpy.linalg.norm(u_b, axis=1)
        # Computation of the sign of the dihedral angle (see: http://structbio.biochem.dal.ca/jrainey/dihedralcalc.html)
        n_b = numpy.cross((frame[a3][1:] - frame[a2][1:]), (frame[a1][:-1] - frame[a2][1:]))
        sign = -numpy.sign(numpy.einsum('ij,ji->i', n_b, (frame[b2][1:] - frame[b3][1:]).T))
        phi = sign * numpy.arccos(dotp / (norm_u_a * norm_u_b))
        return phi

    def get_psi(self, frame_id):
        """

        :param frame_id: the id of the frame to compute the psi angle on
        :type frame_id: int
        :return: the psi dihedral angle in radian for the frame given by frame id
        """
        atomnames = self.traj.struct.atoms['atomname']
        a1 = atomnames == 'N'  # the notation come from: http://en.wikipedia.org/wiki/Dihedral_angle
        a2 = atomnames == 'CA'
        a3 = atomnames == 'C'
        frame = self.traj.array[frame_id]
        b1 = a2
        b2 = a3
        b3 = a1
        u_a = numpy.cross((frame[a2][:-1] - frame[a1][:-1]), (frame[a3][:-1] - frame[a1][:-1]))
        u_b = numpy.cross((frame[b2][:-1] - frame[b1][:-1]), (frame[b3][1:] - frame[b1][:-1]))
        # below is the dot product only for the diagonal (see: http://stackoverflow.com/a/14759341/1679629)
        dotp = numpy.einsum('ij,ji->i', u_a, u_b.T)
        norm_u_a = numpy.linalg.norm(u_a, axis=1)
        norm_u_b = numpy.linalg.norm(u_b, axis=1)
        # Computation of the sign of the dihedral angle (see: http://structbio.biochem.dal.ca/jrainey/dihedralcalc.html)
        n_b = numpy.cross((frame[a3][1:] - frame[a2][1:]), (frame[a1][:-1] - frame[a2][1:]))
        sign = numpy.sign(numpy.einsum('ij,ji->i', n_b, (frame[b2][1:] - frame[b3][1:]).T))
        psi = sign * numpy.arccos(dotp / (norm_u_a * norm_u_b))
        return psi

    @property
    def dihedral_descriptors(self):
        """

        :return: the dihedral angles (phi, psi) for each frames in complex number (cos(phi) + i*sin(phi))
        """
        descriptors = []
        for frame_id in range(self.traj.nframe):
            psi = self.get_psi(frame_id)
            phi = self.get_phi(frame_id)
            psi_complex = numpy.cos(psi) + 1j * numpy.sin(psi)
            phi_complex = numpy.cos(phi) + 1j * numpy.sin(phi)
            descriptor = numpy.asarray(zip(phi_complex[:-1],psi_complex[1:])).flatten()
            descriptors.append(descriptor)
        return numpy.asarray(descriptors)