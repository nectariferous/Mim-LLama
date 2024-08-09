import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading

class AutoCommitHandler(FileSystemEventHandler):
    def __init__(self):
        self.commit_in_progress = False
        self.debounce_timer = None

    def on_any_event(self, event):
        if event.is_directory:
            return
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(5, self.commit_changes)
        self.debounce_timer.start()

    def commit_changes(self):
        if not self.commit_in_progress:
            self.commit_in_progress = True
            subprocess.run(['git', 'add', '.'])
            commit_message = f"Auto-commit: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(['git', 'commit', '-m', commit_message])
            self.commit_in_progress = False

if __name__ == "__main__":
    path = "."  # Monitor the current directory
    event_handler = AutoCommitHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print("Auto-commit script started. Monitoring for changes...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Auto-commit script stopped.")

    observer.join()
