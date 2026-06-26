from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "image_enhancement_cpp",
        [
            "image_enhancement.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        libraries=["opencv_core", "opencv_imgproc", "opencv_highgui"],
        extra_compile_args=["-std=c++11", "-O3"],
        extra_link_args=["-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui"],
    ),
]

setup(
    name="image_enhancement_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)