# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# pf3dk is simple. Importing all the spack stuff is overkill.
# from spack import *

# ----------------------------------------------------------------------------
# If you edit this file, save the file and test your package like this:
#
#     spack install pf3dk
#
# You can edit this file again by typing:
#
#     spack edit pf3dk
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------


class Pf3dk(Package):
    """PF3DK contains kernels derived from pF3D and is intended 
       for use in testing compiler optimization. It contains CPU 
       and GPU (OpenMP 4.5) versions of the kernels.
    """

    homepage = "https://github.com/LLNL/PF3DK/"
    url      = "https://github.com/LLNL/PF3DK/"

    # The list of GitHub accounts to
    # notify when the package is updated.
    maintainers = ['shlanger']

    version('1.0', sha256='77678ec5a53947ead253e5f7ff1f869f85c798e9f20eadb220a276df5d40195e')

    # depends_on('mpi')

    def install(self, spec, prefix):
        make('pf3dtest')
