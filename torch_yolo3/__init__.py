import os

import matplotlib

# in case you are running on machine without display, e.g. server
if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
