"""Reads GSO parameters from configuration file"""

import os
from pathlib import Path
from configparser import ConfigParser
from lightdock.error.lightdock_errors import GSOParameteresError


class GSOParameters(object):
    """Represents the set of the variables of the algorithm"""

    def __init__(self, file_name=None):
        self._config = ConfigParser()
        try:
            if file_name:
                self._config.readfp(open(file_name))
            else:
                self._config.readfp(
                    open(Path(os.environ["LIGHTDOCK_CONF_PATH"]) / "glowworm.conf")
                )
        except Exception as e:
            raise GSOParameteresError(str(e))

        try:
            self.rho = 0.0
            self.gamma = 0.0
            self.beta = 0.0
            self.initial_luciferin = 0.0
            self.initial_vision_range = 0.0
            self.max_vision_range = 0.0
            self.max_neighbors = 0

            self.rmax = int(self._config.get("TTPY", "rmax"))
            self.ngrid = int(self._config.get("TTPY", "ngrid"))

        except Exception as e:
            raise GSOParameteresError(
                "Problem parsing GSO parameters file. Details: %s" % str(e)
            )
