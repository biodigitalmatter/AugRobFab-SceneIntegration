
# Scene Integration Module

For complete system functionality, refer to the [Augmented Robotic Fabrication](https://github.com/AugmentedRoboticFabrication) repository.
## Installation

### Major Requirements

- **Visual Studio 2019 (v16.11)**: Install with the 'Desktop development with C++' option, including MSVC VS 2015 and MSVC VS 2017 components.
- **CMake (v3.28.0-rc5)**: Ensure it is added to the system PATH during installation for easier command line access.
- **NVIDIA Driver (v546.17)**: Required for GPU acceleration.
- **CUDA Toolkit (v11.8.0)**: Essential for leveraging NVIDIA GPU capabilities.
- **Anaconda (v2023.09)**: For managing Python environments and dependencies.
- **Microsoft Azure Kinect SDK (v1.4.1)**: For interfacing with the Azure Kinect sensor.

### Setting Up the Conda Environment

Create a dedicated conda environment using Python 3.9:
```
conda create --name augrobfab python=3.9
```

### Open3D Installation for Enhanced 3D Processing

1. Clone the Open3D repository (v0.17.0), a library essential for 3D data processing:
   ```
   git clone https://github.com/isl-org/Open3D.git -b v0.17.0
   cd <Path-To-Open3D-Directory>
   ```

2. Build Open3D from source, enabling CUDA support for high-performance computing and ensuring compatibility with the Azure Kinect sensor:
   ```
   mkdir build
   cd build
   cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="<Path-To-Open3D-Directory>" -DBUILD_CUDA_MODULE=OFF -DBUILD_AZURE_KINECT=ON ..
   cmake --build . --config Release --target ALL_BUILD
   cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="<Path-To-Open3D-Directory>" -DBUILD_CUDA_MODULE=ON -DBUILD_AZURE_KINECT=ON ..
   cmake --build . --config Release --target ALL_BUILD
   cmake --build . --config Release --target pip-package
   ```

3. Install the compiled Open3D Python package within the conda environment to integrate it with your Python projects:
   ```
   conda activate augrobfab
   cd lib\python_package\pip_package
   pip install open3d*.whl
   python -c 'import open3d as o3d; print(o3d.core.cuda.is_available())'
   ```

### Installing Additional Python Dependencies

Navigate to the project repository and install the necessary Python packages:
```
cd <Path-To-Scene-Integration-Repo>
pip install '.[kinect]'
```

Running `pip install .` will skip the Azure Kinect library [`pyk4a`](https://pypi.org/project/pyk4a/).

## Citation

Please cite [our work](https://doi.org/10.1007/s41693-022-00081-4) if you use our codebase in your research.

```bib
@article{Capunaman2022,
  author = {Çapunaman, Özgüç Bertuğ and Dong, Wei and Gürsoy, Benay},
  title = {A Vision-Based Sensing Framework for Adaptive Robotic Tooling of Indefinite Surfaces},
  journal = {Construction Robotics},
  year = {2022},
  doi = {10.1007/s41693-022-00081-4},
}
```
