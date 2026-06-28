DEFAULT_EVENT_START_SECONDS = 3 * 24 * 60 * 60
DEFAULT_EVENT_END_SECONDS = 4 * 24 * 60 * 60
DEFAULT_HISTORY = 5

#How where these constants chosen?
# event start/end: taken from the simulated contamination schedule
# history: chosen as a simple initial lag length for the ANN, then kept as the default for now