from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='glut_cuda',
    ext_modules=[
        CUDAExtension(
            name='glut_cuda',
            sources=[
                'glut_binding.cpp',
                'glut_cuda.cu',
            ],
            extra_compile_args={
                'cxx' : ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',          # fast intrinsics: __expf, __logf, ...
                    '-arch=sm_89',              # RTX 4090; use sm_86 for RTX 3090, sm_80 for A100
                    '--ptxas-options=-v',       # print register usage
                    '-maxrregcount=64',         # cap registers to improve occupancy
                ],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)