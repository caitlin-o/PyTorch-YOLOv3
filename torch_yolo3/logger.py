import os
import time

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Logger.

    >>> log = Logger('.')
    >>> log.list_scalars_summary([('loss', 0.5)], 0)
    """

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self._log_dir = log_dir
        self._run_datetime = time.strftime("%Y-%m-%d_%H:%M")
        self.writers = {
            k: SummaryWriter(log_dir=os.path.join(self._log_dir, self._run_datetime, k))
            for k in ('train', 'val')
        }

    def scalar_summary(self, tag, value, step, phase='train'):
        """Log a scalar variable."""
        assert phase in self.writers
        self.writers[phase].add_scalar(tag, value, step)

    def list_scalars_summary(self, tag_value_pairs, step, phase='train'):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.scalar_summary(tag, value, step, phase)

    def __del__(self):
        for _, writer in self.writers.items():
            writer.flush()
            writer.close()
