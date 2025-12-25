
import sys
import os
import datetime

class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        # Ensure log directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        # Only add timestamp if message is not just a newline
        if message.strip():
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.terminal.write(message)
            self.log.write(timestamp + message)
        else:
            self.terminal.write(message)
            self.log.write(message)
            
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(target_name, condition, prefix="Train"):
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Filename: logs/Train_DryWeight_Light_20231224_223000.log
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{prefix}_{target_name}_{condition}_{now_str}.log"
    
    # Redirect stdout
    sys.stdout = Logger(log_filename)
    print(f"--> Logging to: {log_filename}")
    return log_filename
