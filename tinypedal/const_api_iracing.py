"""iRacing API constants

Defines API identifier and configuration for iRacing integration
"""

API_NAME = "iRacing"
API_SHORTNAME = "IRA"
API_ENABLED = True

# Module paths
CONNECTOR_CLASS = "iracing_connector.IRacingConnector"
READER_CLASS = "iracing_reader.IRacingReader"

# Configuration
UPDATE_INTERVAL = 0.01  # 10ms
RECONNECT_INTERVAL = 1.0  # 1 second
READER_STOP_TIMEOUT = 2.0  # 2 seconds
