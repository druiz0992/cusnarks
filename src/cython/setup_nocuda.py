"""
/*
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : setup.py
//
// Date       : 04/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Setup functionality to build Cython wrappers
//
// NOTES
//    File based from https://github.com/rmcgibbo/npcuda-example/tree/master/cython/setup.py
//
// -----------------------------------------------------------------

"""

# from future.utils import iteritems
import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext, Extension
from Cython.Build import cythonize
import numpy


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


CUDA = None

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


if CUDA is not None:
  ext = [ Extension('pycusnarks',
                    cython_compile_time_env=dict(CUDA_DEF=True),
                    sources = ['_cusnarks_kernel.pyx'],
                    library_dirs = [CUDA['lib64']],
                    libraries = ['cudart','cusnarks'],
                    language = 'c++',
                    runtime_library_dirs = [CUDA['lib64']],
                     # This syntax is specific to this build system
                     # we're only going to use certain compiler args with nvcc
                     # and not with gcc the implementation of this trick is in
                     # customize_compiler()
                     extra_compile_args= {
                         'gcc': [],
                         'nvcc': [
                         '-arch=sm_60', '--ptxas-options=-v', '-c',
                         '--compiler-options', "'-fPIC'"
                        ]
                      },
                    include_dirs = [numpy_include, CUDA['include'], '../cuda'] 
                 )
      ]
  cmdc = {'build_ext': custom_build_ext}
else:
  ext = [ Extension('pycusnarks', 
                    cython_compile_time_env=dict(CUDA_DEF=False,),
                    sources = ['_cusnarks_kernel.pyx'],
                    libraries = ['cusnarks'],
                    language = 'c++',
                    include_dirs = [numpy_include, '../cuda'] 
                 )
      ]
  cmdc = {'build_ext': build_ext}



setup(name = 'cusnarks-lib',
      # Random metadata. there's more you can supply
      author = 'David Ruiz',
      version = '0.1',

      ext_modules = ext,
      #ext_modules = cythonize(ext),

      # Inject our custom trigger
      #cmdclass = {'build_ext': custom_build_ext}
      #cmdclass = {'build_ext': build_ext}
      cmdclass = cmdc

      # Since the package has c code, the egg cannot be zipped
      #zip_safe = False
      )


