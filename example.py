import numpy as np
from scenredpy import Class_scenred

sr_instance = Class_scenred()

sr_instance.import_data("test_data.h5")
sr_instance.prepare_data()

#sr_instance.scenario_reduction(dist_type="cityblock", fix_node=1, tol_node=np.linspace(1,24, 24)) #fix_prob_tol
sr_instance.scenario_reduction(dist_type="cityblock",fix_prob=1, tol_prob=np.linspace(0, 0.2, 24))

sr_instance.draw_red_scenario()

sr_instance.sort_result()







