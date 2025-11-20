import numpy as np
from .mcaploader import McapLoader
import scipy.interpolate as si
import scipy.spatial.transform as st

def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp

class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_quat(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_quat()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose

def remove_duplicate_timestamps(t, y):
    if len(t) != len(y):
        raise ValueError
    
    _, unique_indices = np.unique(t, return_index=True)
    
    unique_indices = np.sort(unique_indices)
    
    unique_t = t[unique_indices]
    unique_y = y[unique_indices]
    
    return unique_t, unique_y


def get_inter_data(
    bag: McapLoader, topic_name: str, ref_timestamp: list, inter_type: str
) -> np.ndarray:
    assert inter_type in ["linear", "pose"]
    topic_data = bag.get_topic_data(topic_name)
    topic_data_array = np.array([d["decode_data"] for d in topic_data])
    topic_ts = np.array([d["data"].header.timestamp for d in topic_data])

    # remove duplicate timestamp
    # sometimes high frequency data may have this issue (ie. imu data)
    clean_topic_ts, clean_topic_data_array = remove_duplicate_timestamps(
        topic_ts, topic_data_array
    )
    if clean_topic_data_array.shape[0] != topic_data_array.shape[0]:
        print(f"find uplicate timestamps in {topic_name}")

    assert topic_data_array.ndim == 2
    if inter_type == "linear":
        data_inter = get_interp1d(t=clean_topic_ts, x=clean_topic_data_array)
        topic_data_inter = data_inter(ref_timestamp).astype(np.float32)
    elif inter_type == "pose":
        data_inter = PoseInterpolator(t=clean_topic_ts, x=clean_topic_data_array)
        topic_data_inter = data_inter(ref_timestamp).astype(np.float32)

    return topic_data_inter