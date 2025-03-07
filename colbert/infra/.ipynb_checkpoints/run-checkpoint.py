import os
import atexit
import subprocess
import socket
from pathlib import Path
from threading import Thread
import time

from colbert.utils.utils import create_directory, print_message, timestamp
from contextlib import contextmanager

from colbert.infra.config import RunConfig


class Run(object):
    _instance = None

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # NOTE: If a deadlock arises, switch to false!!

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []
            cls._instance.time_stamp_to_debug = time.time()
            cls._instance.tracker = None  # Initialize tracker reference
            cls._instance.tensorboard_process = None  # Initialize TensorBoard process reference

            # TODO: Save a timestamp here! And re-use it! But allow the user to override it on calling Run().context a second time.
            run_config = RunConfig()
            run_config.assign_defaults()
            
            cls._instance.__append(run_config)

        # TODO: atexit.register(all_done)

        return cls._instance

    @property
    def config(self):
        return self.stack[-1]

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        super().__getattr__(name)

    def __append(self, runconfig: RunConfig):
        # runconfig.disallow_writes(readonly=True)
        self.stack.append(runconfig)

    def __pop(self):
        self.stack.pop()

    @contextmanager
    def context(self, runconfig: RunConfig, inherit_config=True):
        if inherit_config:
            runconfig = RunConfig.from_existing(self.config, runconfig)

        self.__append(runconfig)

        try:
            yield
        finally:
            self.__pop()
        
    def open(self, path, mode='r'):
        path = os.path.join(self.path_, path)

        if not os.path.exists(self.path_):
            create_directory(self.path_)

        if ('w' in mode or 'a' in mode) and not self.overwrite:
            assert not os.path.exists(path), (self.overwrite, path)

            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

        return open(path, mode=mode)
    
    def print(self, *args):
        print_message("[" + str(self.rank) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.rank == 0:
            self.print(*args)

    # === NEW METHODS FOR TRACKING ===
    
    def set_tracker(self, tracker, keep_tensorboard_running=False):
        """
        Set a tracker instance to be used for logging metrics and optionally start TensorBoard
        
        Args:
            tracker: The tracker instance
            keep_tensorboard_running: If True, TensorBoard will keep running after training completes
                                     If False (default), TensorBoard will be terminated when the script exits
        """
        print("Run class time stamp", self.time_stamp_to_debug)
        self.tracker = tracker
        self.keep_tensorboard_running = keep_tensorboard_running  # Store the preference
        
        # If tracker has TensorBoard enabled, start a TensorBoard server
        if hasattr(tracker, 'enable_tensorboard') and tracker.enable_tensorboard and self.rank == 0:
            self._start_tensorboard(Path(tracker.log_dir))
            
        return self
    
    def _start_tensorboard(self, log_dir, port=6006):
        """Start a TensorBoard server for the current run"""
        # Find available port
        while not self._is_port_available(port):
            port += 1
        
        # Start TensorBoard in a separate thread
        def run_tensorboard():
            # If keeping tensorboard running, use start_new_session=True to detach the process
            if getattr(self, 'keep_tensorboard_running', False):
                self.tensorboard_process = subprocess.Popen(
                    ["tensorboard", "--logdir", str(log_dir), "--port", str(port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True  # This detaches the process
                )
            else:
                self.tensorboard_process = subprocess.Popen(
                    ["tensorboard", "--logdir", str(log_dir), "--port", str(port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        
        thread = Thread(target=run_tensorboard)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for TensorBoard to initialize
        time.sleep(2)
        
        # Try to get the IP address with better error handling
        hostname = socket.gethostname()
        try:
            # Try to get the IP address
            ip_address = socket.gethostbyname(hostname)
            tensorboard_url = f"http://{ip_address}:{port}"
        except socket.gaierror:
            # Fallback to localhost if hostname resolution fails
            print_message("Could not resolve hostname. Using localhost for TensorBoard URL.")
            tensorboard_url = f"http://localhost:{port}"
            
            # Try an alternative approach to get an IP address
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                alternative_ip = s.getsockname()[0]
                s.close()
                print_message(f"Alternative network accessible URL: http://{alternative_ip}:{port}")
                tensorboard_url_alt = f"http://{alternative_ip}:{port}"
            except:
                tensorboard_url_alt = None
        
        # Create and save the URL(s)
        url_file = log_dir / "tensorboard_url.txt"
        with open(url_file, "w") as f:
            f.write(f"TensorBoard URL: {tensorboard_url}\n")
            if 'tensorboard_url_alt' in locals() and tensorboard_url_alt:
                f.write(f"Alternative URL: {tensorboard_url_alt}\n")
        
        print_message("=" * 80)
        print_message(f"TensorBoard running at: {tensorboard_url}")
        if 'tensorboard_url_alt' in locals() and tensorboard_url_alt:
            print_message(f"Alternative URL: {tensorboard_url_alt}")
        
        # Display warning if TensorBoard will keep running
        if getattr(self, 'keep_tensorboard_running', False):
            print_message("WARNING: TensorBoard will continue running after training completes.")
            print_message("         Port will remain in use until the TensorBoard process is manually terminated.")
        print_message("=" * 80)
        
        # Store the URL and process in the Run instance
        self.tensorboard_url = tensorboard_url
        
        # Register cleanup only if we don't want to keep TensorBoard running
        if not getattr(self, 'keep_tensorboard_running', False):
            atexit.register(self._stop_tensorboard)
    
    def _stop_tensorboard(self):
        """Stop the TensorBoard server if it's running"""
        if hasattr(self, 'tensorboard_process') and self.tensorboard_process:
            self.tensorboard_process.terminate()
            self.tensorboard_process = None
    
    def _is_port_available(self, port):
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    # Add method to properly clean up resources
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'tensorboard_process'):
            self._stop_tensorboard()
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.close()
            
    def log_metric(self, name, value, step=None, **kwargs):
        """
        Log a metric to the tracker if available
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (optional)
            **kwargs: Additional arguments to pass to the tracker
        """
        print(f'logging in the Run {Run().rank}, {self.tracker}')
        print("Run class time stamp in log new instance", Run().time_stamp_to_debug)
        print("Run class time stamp in log old instance", self.time_stamp_to_debug)
        
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.log_metric(name, value, step, **kwargs)
        else:
            # Fallback behavior - just print the metric
            self.print_main(f"Metric {name} = {value}" + (f" (step {step})" if step is not None else ""))
            
    def log_metrics(self, metrics_dict, step=None, **kwargs):
        """
        Log multiple metrics at once
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step number (optional)
            **kwargs: Additional arguments to pass to the tracker
        """
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.log_metrics(metrics_dict, step, **kwargs)
        else:
            # Fallback behavior - log each metric individually
            for name, value in metrics_dict.items():
                self.log_metric(name, value, step, **kwargs)
    
    def add_figure(self, tag, figure, step=None):
        """Add a matplotlib figure to the tracker"""
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.add_figure(tag, figure, step)
    
    def add_text(self, tag, text, step=None):
        """Add text to the tracker"""
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.add_text(tag, text, step)
            
    # === END NEW METHODS ===


if __name__ == '__main__':
    print(Run().root, '!')

    with Run().context(RunConfig(rank=0, nranks=1)):
        with Run().context(RunConfig(experiment='newproject')):
            print(Run().nranks, '!')

        print(Run().config, '!')
        print(Run().rank)


# TODO: Handle logging all prints to a file. There should be a way to determine the level of logs that go to stdout.