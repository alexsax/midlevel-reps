import json
import os
from sacred.observers import FileStorageObserver

class FileStorageObserverWithExUuid(FileStorageObserver):
    ''' Wraps the FileStorageObserver so that we can pass in the Id.
        This allows us to save experiments into subdirectories with 
        meaningful names. The standard FileStorageObserver jsut increments 
        a counter.'''
    
    UNUSED_VALUE = -1
    
    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        _id = config['uuid'] + "_metadata"
        super().started_event(ex_info, command, host_info, start_time, config,
                      meta_info, _id=_id)

    def queued_event(self, ex_info, command, host_info, queue_time, config,
                         meta_info, _id):
        assert 'uuid' in config, "The config must contain a key 'uuid'"
        _id = config['uuid'] + "_metadata"
        super().queued_event(ex_info, command, host_info, queue_time, config,
                         meta_info, _id=_id)