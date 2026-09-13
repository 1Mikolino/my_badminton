import yaml
import sys
from pathlib import Path
sys.path.append(r'F:\my_badminton\src\simulation')
import config
print(config.load_yaml(r"F:\my_badminton\config\hysics.yaml"))