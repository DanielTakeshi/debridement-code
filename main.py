from dvrk.robot import *
from autolab.data_collector import *

import time


d = DataCollector()

time.sleep(2)


psm1 = robot("PSM1")

psm1.home()

