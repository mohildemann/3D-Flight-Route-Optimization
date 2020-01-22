# 3DRouting
This is a short description of my Master Thesis, a 3D route optimization for air taxis with the multiple criteria "Shortest Flight Time", "Lowest Energy Consumption" and "Least Added Noise"

Requirements:
1. Install and get license for ArcGIS Pro (https://pro.arcgis.com/en/pro-app/get-started/install-and-sign-in-to-arcgis-pro.htm)
2. Set up your Python environment: You need Python 3.68 (Comes with ArcGIS Pro) 
3. Use the environment of ArcGIS Pro to run the main.py. For example, the environment for my installation is: "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
4. Download the Geodatabase "OptimizationInputs"
5. In init.py, set arcpy.env.workspace = r'...\OptimizationInputs.gdb' to the path where you stored the downloaded Geodatabase

Running the optimization:
1. Define, which aircraft you want to use. Currently you can use the flight characteristics of the Lilium Jet 5-seater (aircraft = "Lilium") or to the Ehang 184 (aircraft = "EHANG") in main.py
2. Define the parameters that you want to use in main.py
3. Define how often you want to run the loop in the main.py at location "for seed in range(1):"
4. Run the main and do not open ArcGIS Pro until optimization has finished. Otherwise you can lock the Geodatabase in use.

Getting the results:
1. The logged data that contain the fitness values are stored in the directory "LogFiles". The runs from you contain "new" in their names.
2. The Point Feature Classes are stored in the Geodatabase.
