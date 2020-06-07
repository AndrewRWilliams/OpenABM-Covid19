import re
import string

# Utilities
URL_REGEX = re.compile(r"^(https?|file)://")

START_OF_TIME = 0
DAYS_IN_A_YEAR = 365.25
WEEKS_IN_A_YEAR = 52.0
