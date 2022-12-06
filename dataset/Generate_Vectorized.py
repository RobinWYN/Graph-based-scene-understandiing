import numpy as np

from l5kit.geometry import compute_agent_pose, rotation33_as_yaw
from l5kit.data import filter_agents_by_labels, PERCEPTION_LABEL_TO_INDEX
from l5kit.data.filter import filter_agents_by_track_id
from l5kit.sampling.agent_sampling import get_agent_context, get_relative_poses


def generate_physical_vectorized(
    state_index, frames, agents, tl_faces, selected_track_id,
    history_num_frames, future_num_frames, filter_agents_threshold, vectorizer,
    max_num_agent, max_num_lanes
    ):

    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        _,
        _,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames, future_num_frames)

    
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    agent_centroid_m = cur_frame["ego_translation"][:2]
    agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)

    future_coords_offset, future_yaws_offset, future_extents, future_availability = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    history_coords_offset, history_yaws_offset, history_extents, history_availability = get_relative_poses(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    history_coords_offset[history_num_frames + 1:] *= 0
    history_yaws_offset[history_num_frames + 1:] *= 0
    history_extents[history_num_frames + 1:] *= 0
    history_availability[history_num_frames + 1:] *= 0

    vectorized_features = vectorizer.vectorize(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                               history_frames, history_agents, history_coords_offset, history_availability)

    all_agents_polyline = np.concatenate(
        [vectorized_features["agent_trajectory_polyline"],
         vectorized_features["other_agents_polyline"],
         vectorized_features["lanes"]],
         axis=0
    )

    all_agents_availability = np.concatenate(
        [vectorized_features["agent_polyline_availability"],
         vectorized_features["other_agents_polyline_availability"],
         vectorized_features["lanes_availabilities"]],
         axis=0
    )

    attention_mask = np.zeros((1 + 2 * max_num_lanes + max_num_agent, ))
    attention_mask[:(vectorized_features["num_agents"] + 1)] = 1
    attention_mask[1 + max_num_agent: (1 + max_num_agent + vectorized_features["num_lane"])] = 1

    future_coords_offset = future_coords_offset.reshape((-1, ))
    target_availabilities = np.zeros((2 * future_num_frames))
    target_availabilities[0::2] = future_availability
    target_availabilities[1::2] = future_availability

    frame_info = {
        "target_positions": future_coords_offset,
        "target_availabilities": target_availabilities,
        "all_agents_polyline":all_agents_polyline,
        "all_agents_availability":all_agents_availability,
        "attention_mask":attention_mask,
        "num_agents":vectorized_features["num_agents"],
    }

    return {**frame_info}