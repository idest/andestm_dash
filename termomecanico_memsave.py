import setup
import compute_memsave
#import plot
import numpy as np
from utils import DotDict
import os
import sys
import resource
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()

def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

print('Leyendo variables...')
t_input = setup.readVars('VarTermal.txt')
m_input = setup.readVars('VarMecanico.txt')
exec_input = setup.readVars('VarExec.txt')
tmc = exec_input.temcaso
mmc = exec_input.meccaso
direTer, direTerMec = setup.makeDirs(exec_input.temcaso, exec_input.meccaso)
gm_data = np.loadtxt('data/Modelo.dat')
areas = np.loadtxt('data/areas.dat')
trench_age = np.loadtxt('data/PuntosFosaEdad.dat')
rhe_data = setup.read_rheo('data/Rhe_Param.dat')
#data_q = np.loadtxt('datos_Q/QS/ObsQs/QsObs.txt')

D, CS, GM, TM, MM = compute_memsave.compute(gm_data, areas, trench_age, rhe_data, t_input, m_input)

x_axis = CS.get_x_axis()
y_axis = CS.get_y_axis()
z_axis = CS.get_z_axis()
xy_step = CS.get_xy_step()
grid_2D = CS.get_2D_grid()
topo = GM.get_topo()
icd = GM.get_icd()
moho = GM.get_moho()
slab_lab = GM.get_slab_lab()
geotherm = TM.get_geotherm()
yse = MM.get_yse()
shf = TM.get_surface_heat_flow()
eet = MM.get_eet()

print("After termomecanico M.S:")
mem()

#plotear perfiles termales
"""
os.chdir(direTer)
fig = plot.plot_thermal(CS.get_axes()[0], CS.get_axes()[2], D, CS, GM, TM)
os.chdir('../../')
"""
"""
#plotear perfiles termomecanicos
os.chdir(direTerMec)
fig = plot.plot_mec(CS.get_axes()[0], CS.get_axes()[2], D, CS, GM, MM)
os.chdir('../../../')
"""
#plotear mapa q_surface
#os.chdir(direTer)
#fig = plot.map_q_surface(CS, TM, tmc, data_q)
#os.chdir('../../')


#detachment = plot.get_detachment(CS,GM,MM)

tracker.print_diff()
