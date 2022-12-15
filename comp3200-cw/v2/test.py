from swarm_distributor import SwarmDistributor
from flask_backend import FlaskBackend
import time
import numpy as np
import logging
from colored_log_formatter import ColoredFormatter

backend_logger = logging.getLogger("backend")
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
backend_logger.addHandler(ch)

backend = FlaskBackend(9001, ["localhost:9001","localhost:9002","localhost:9003"], logger = backend_logger)
dist = SwarmDistributor(np.array([1,2,3]), backend)
dist.update_params(np.array([3,3,3]))
dist.sync()