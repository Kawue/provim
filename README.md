
ProViM
======
ProViM (Basic **Pro**cessing for **V**isualization and Multivariate Analysis of **M**SI Data) is a fast and easy to use preprocessing pipeline for Mass Spectrometry Imaging (MSI) data. The pipeline consists of multiple modules which can be chained together.

## Docker Version
1. Start Docker, navigate into the provim directory and call `docker build -t provim/pipeline .` to build the docker container.
2. Run the docker container with `docker run -v path-to-data:/data --rm provim/pipeline python script.py`.
Using `script.py -h` will provide all necessary information about parameters and functionality of the script.

## General Workflow
The general workflow, with necessary and optional scipts indicated is as follows:
1. `workflow_pybasis.py` (required)
2. `fast_convert_hdf.py` (required)
3. `matrix_preprocessing.py` (optional)
3.1. `interactive_matrix_detection.py` / `automated_matrix_detection.py` (optional, interactive recommended)
3.2. `matrix_postprocessing.py` (optional)
4. `workflow_peakpicking.py` (optional)
5. `msi_image_writer.py` (optional)
While only step one and two are required and the remaining steps can be chained to specific needs, it is recommended to not interchange the order of the scripts and only skip single steps on demand.

The following scripts can also be used stand alone and therefore are easily to combine with other processing pipelines:

a. `workflow_peakpicking.py`

b. `msi_image_writer.py`

c. `msi_dimension_reducer.py`

If the existing file type is imzML instead of HDF5 we refer to https://github.com/Kawue/imzML-to-HDF5.

## Local Version (without Docker)
1. Install Anaconda.
2. Install the environment with the `environment.yml` by calling `conda env create -f environment.yml`.
3. Activate the environment with `conda active provim` and follow the **General Workflow**.
4. Call the scripts manually.

## Docker on Windows 7, 8 and 10 Home
1. Visit https://docs.docker.com/toolbox/toolbox_install_windows/. Follow the given instructions and install all required software to install the Docker Toolbox on Windows.
2. Control if virtualization is enabled on your system. Task Manager -> Performance tab -> CPU -> Virtualization. If it is enabled continue with Step X.
3. If virtualization is disabled, it needs to be enabled in your BIOS. Navigate into your systems BIOS and look for Virtualization Technology (VT). Enable VT, save settings and reboot. This option is most likely part of the Advanced or Security tab. This step can deviate based on your Windows and Mainboard Manufacturer.
4. Open your CMD as administrator and call `docker-machine create default --virtualbox-no-vtx-check`. A restart may be required.
5. In your OracleVM VirtualBox selected the appropriate machine (probably the one labeled "default") -> Settings -> Network -> Adapter 1 -> Advanced -> Port Forwarding. Click on "+" to add a new rule and set Host Port to 8080 and Guest Port to 8080. Be sure to leave Host IP and Guest IP empty. Also, add another rule for the Port 5000 in the same way. A restart of your VirtualBox may be needed.
6. Now you should be ready to use the Docker QuickStart Shell to call the Docker commands provided to start this tool.
