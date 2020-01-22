import arcpy
from arcpy import env
from arcpy.sa import *
arcpy.env.outputZFlag = "Enabled"
arcpy.CheckOutExtension('Spatial')
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
arcpy.env.overwriteOutput = True
arcpy.env.workspace = r'D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\OptimizationInputs.gdb'

def aircraft_specs(aircraft):
    if aircraft == "Lilium":
        #This setup: (Lilium Jet, Electric VTOL Configurations Comparison 2018)
        type_aircraft = "vectored thrust eVTOL"
        weight_aircraft =  490 #(in kg)
        wing_area = 3.6 #(in m². Calculated with a wingspan of 6, Root chord 78 cm of and Tip chord of 42 cm)
        CD_from_drag_polar = (lambda CL: 0.0163 + 0.058 ** CL) #CD = Drag coefficicient, CL = Lift coefficient of plane. Obtained formula: CD = 0.0163 + 0.058 * CL²
        maximum_speed_air_taxi = 70  # (in m/s, 252 in km/h)
        acceleration_speed = 2  # (in m/s²)
        acceleration_energy = 187  # (in kW)
        deceleration_speed = -2  # (in m/s²)
        deceleration_energy = 187  # (in kW)
        minimal_cruise_energy = 28 # (in kW at speed with perfect lift/drag ratio)
        take_off_and_landing_energy = 187
        hover_energy = 187 # (in kW)
        noise_pressure_acceleration = 100
        noise_pressure_deceleration = 100
        noise_at_cruise = 100
        noise_at_hover = 100
    else:
        # This setup: (Ehang, Multicoptor Configurations Comparison 2018)
        type_aircraft = "multicoptor"
        weight_aircraft =  260 #(in kg)
        wing_area = 0 #(in m². Calculated with a wingspan of 6, Root chord 78 cm of and Tip chord of 42 cm)
        CD_from_drag_polar = None #CD = Drag coefficicient, CL = Lift coefficient of plane. Obtained formula: CD = 0.0163 + 0.058 * CL²
        maximum_speed_air_taxi = 27.777  # (in m/s, 252 in km/h)
        acceleration_speed = 2  # (in m/s²)
        acceleration_energy = 42.1  # (in kW)
        deceleration_speed = -2  # (in m/s²)
        deceleration_energy = 42.1  # (in kW)
        minimal_cruise_energy = 42.1 # (in kW at speed with perfect lift/drag ratio)
        take_off_and_landing_energy = 42.1
        hover_energy = 42.1 # (in kW)
        noise_pressure_acceleration = 100
        noise_pressure_deceleration = 100
        noise_at_cruise = 100
        noise_at_hover = 100
    return type_aircraft,weight_aircraft,wing_area,CD_from_drag_polar,maximum_speed_air_taxi,acceleration_speed,acceleration_energy,deceleration_speed,deceleration_energy,minimal_cruise_energy,take_off_and_landing_energy,hover_energy,noise_pressure_acceleration, noise_pressure_deceleration, noise_at_cruise, noise_at_hover
