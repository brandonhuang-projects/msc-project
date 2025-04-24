# shared_memory_processor/processor.py
import sys, os, json, uuid, random
from datetime import datetime
import numpy as np
from .utils import get_script_dir, should_serialize

current_dir = get_script_dir()
shared_memory_dir = os.path.join(os.path.dirname(current_dir), "shared_memory-main", "python")
sys.path.insert(0, shared_memory_dir)

# Import shared memory operations
from shared_memory import set_shared_memory_path, get_shared_memory_data, set_shared_memory_data

class SHMEM_Processor:
    """
   Processor for managing shared memory segments.
   
   Attributes:
       BASE_DIR (str): The base directory of this module.
       BUFFER_PATH (str): Directory path for IPC buffer.
       METADATA_PATH (str): File path for shared memory metadata.
   """
    # Set the working directory relative to this file.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUFFER_PATH = os.path.join(BASE_DIR, "__ipc_buffer__")
    METADATA_PATH = os.path.join(BUFFER_PATH, "_metadata.json")
    
    def __init__(self):
        """
        Initialize the processor with a unique session ID and clear existing variables.
        """
        self._session_id = str(uuid.uuid4())
        self.clear_vars()
        # DEBUGGING: change to False to retain input data
        self.OVERWRITE = True
    
    def clear_vars(self):
        """
        Clear existing shared memory variables by deleting all files in the buffer directory.
        Resets metadata to include current session ID and an empty segments list.
        """
        # Create buffer directory if it doesn't exist
        if not os.path.exists(self.BUFFER_PATH):
            os.makedirs(self.BUFFER_PATH)
            
        # Remove all files and subdirectories within the buffer
        for file in os.listdir(self.BUFFER_PATH):
            file_path = os.path.join(self.BUFFER_PATH, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or link
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)  # Remove directory
                
        # Reset metadata file with a new session ID and empty segments
        with open(self.METADATA_PATH, 'w') as f:
            json.dump({"session_id": self._session_id, "segments": []}, f)
    
    def _read_shared_memory_metadata(self):
        """
        Read metadata from the shared memory metadata file and load associated data segments.
        
        Returns:
            list: A list of data segments loaded from shared memory.
        """
        with open(self.METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        segments = metadata.get('segments', [])
        args = []
        
        # Loop through segments and load shared memory data for valid segment files
        for seg in segments:
            
            if seg.get("inline", False):
                # Inline data stored directly in metadata.
                data = seg.get("inlineData")
                # If data was serialized as JSON, decode it.
                if seg.get("dataType") == "json":
                    data = json.loads(data)
                args.append(data)
                continue
            
            file_path = seg.get("filePath")
            if not file_path.endswith(".shmem"):
                continue  # Skip non-shared memory files
            set_shared_memory_path(file_path)
            data = get_shared_memory_data()
            if seg.get("dataType") == "json":
                # Deserialize complex data.
                data = json.loads(data)
            else:
                if type(data) != str:
                    data = data.squeeze()
            args.append(data)
        return args
    
    def _create_shared_memory_segments(self, *args):
        """
        Overwrite existing metadata with new shared-memory segments for each argument.
        
        Args:
            *args: Variable length argument list to be stored as shared memory segments.
        
        Returns:
            dict: New metadata containing session ID and the freshly written segments.
        """
        buffer_path = self.BUFFER_PATH
        meta_path = self.METADATA_PATH
        OVERWRITE = self.OVERWRITE
        if not os.path.exists(buffer_path):
            os.makedirs(buffer_path)
            
        # Load existing metadata if it exists
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {'session_id': self._session_id, 'segments': []}
        session_id = existing_metadata.get('session_id', self._session_id)
        existing_segments = {seg['varId']: seg for seg in existing_metadata.get('segments', [])}
        if OVERWRITE:
            existing_segments_overwrite = {
                seg['varId']: seg
                for seg in existing_metadata.get('segments', [])
                if not seg.get('inline', False)
            }
            existing_segments = {}
            
        # Create a new shared memory segment for each argument
        for value in args:
            var_id = '{:08x}'.format(random.getrandbits(32))
            
            if (isinstance(value, np.ndarray) and value.size == 1) or isinstance(value, np.generic):
                value = value.item()    
            
            if isinstance(value, (int, float, str, bool, bytes)):
                segment_entry = {
                    'varId': var_id,
                    'source': 'Python',
                    'inline': True,
                    'inlineData': value,
                    'dataType': type(value).__name__,
                    'dimensions': [],
                    'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                }
                existing_segments[var_id] = segment_entry
                continue  # Skip creating a shared memory file.
            
            if OVERWRITE:  
                if existing_segments_overwrite:
                    var_id = existing_segments_overwrite.popitem()[0]
                
            segment_file = os.path.join(buffer_path, var_id + ".shmem")
            set_shared_memory_path(segment_file)
            
            if should_serialize(value):
                # For non-uniform data types, use JSON serialization.
               data_type = "json"
               dimensions = len(value)
               value = json.dumps(value)
            
            else:
                # Determine dimensions and data type based on value type
                if isinstance(value, np.ndarray):
                    dimensions = list(value.shape)
                    data_type = str(value.dtype)
                else:
                    try:
                        dimensions = [len(value)]
                    except Exception:
                        dimensions = []
                    data_type = type(value).__name__
                
            set_shared_memory_data(value)
                
            # Add segment info to existing segments dictionary
            existing_segments[var_id] = {
                'varId': var_id,
                'source': 'Python',
                'filePath': segment_file,
                'dataType': data_type,
                'dimensions': dimensions,
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            }
            
        # Update metadata with new segments
        updated_metadata = {
            'session_id': session_id,
            'segments': list(existing_segments.values())
        }
        
        # Write updated metadata back to file
        with open(meta_path, 'w') as f:
            json.dump(updated_metadata, f, indent=4)
        return updated_metadata
    
    def call(self, func):
        """
        Call a function wrapped with shared memory synchronization using this processor.
        
        Args:
            func (callable): Function to be synchronized and executed.
        """
        # Import the sync decorator locally to avoid circular imports
        from shared_memory_processor.decorators import sync_shared_memory
        decorated = sync_shared_memory(self, func)
        decorated()
