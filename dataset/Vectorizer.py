import numpy as np

from l5kit.data import filter_agents_by_distance, filter_agents_by_labels
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.geometry.transform import transform_points
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.sampling.agent_sampling import get_relative_poses

TYPE_INDEX = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0]

class PhysicalVectorizer:
    def __init__(self, mapAPI, cfg):
        """
        Generate trajectory and lane information

        Args:
            mapAPI : map information
            cfg: dict
        """
        self.mapAPI = mapAPI
        self.cfg = cfg
        self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
        self.history_num_frames = cfg["model_params"]["history_num_frames"]
        self.max_num_agent = cfg["data_generation_params"]["max_num_agent"]
        self.max_num_lanes = cfg["data_generation_params"]["max_num_lanes"]

    def get_poly(self, agent_history_coords_offset):

        agent_history_start = agent_history_coords_offset[:-1]
        agent_history_end = agent_history_coords_offset[1:]
        return np.concatenate([agent_history_start, agent_history_end], axis=1)

    def vectorize(self, selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world, 
                  history_frames, history_agents, history_position_m, history_availability):
        """
        Get agent features and map features respectively
        """

        agent_features = self._vectorize_agents(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                                history_frames, history_agents, history_position_m, history_availability)
        map_features = self._vectorize_map(agent_centroid_m, agent_from_world)

        return {**agent_features, **map_features}

    def _vectorize_agents(self, selected_track_id, agent_centroid_m, agent_yaw_rad, 
                          agent_from_world, history_frames, history_agents, 
                          history_position_m, history_availability):


        agent_history_timestamp = np.arange(self.history_num_frames).reshape(-1, 1)
        num_agents = 0

        agent_trajectory_polyline = np.zeros((1, self.history_num_frames, 7))
        agent_polyline_availability = np.zeros((1, self.history_num_frames))
        
        agent_trajectory_polyline[0, :, :] = np.concatenate(
                [self.get_poly(history_position_m[::-1][:]),
                 agent_history_timestamp, 
                 np.zeros((self.history_num_frames, 1)), 
                 np.ones((self.history_num_frames, 1))],
                axis=1
        )

        history_availability = history_availability[::-1]

        agent_polyline_availability[0] = np.logical_and(
            history_availability[:-1],
            history_availability[1:],
        )

        # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, self.max_agents_distance)

        cur_agents = filter_agents_by_labels(
            history_agents[0], threshold=self.cfg["model_params"]["filter_agents_threshold"]
        )
        cur_agents = filter_agents_by_distance(
            cur_agents, agent_centroid_m, self.max_agents_distance
        )

        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, max_agents=150
        )

        all_other_agents_history_positions = np.zeros((self.max_num_agent, self.history_num_frames, 7))
        all_other_agents_history_availability = np.zeros((self.max_num_agent, self.history_num_frames))

        for idx, track_id in enumerate(list_agents_to_take):
            
            if idx == self.max_num_agent:
                break

            (
                agent_history_coords_offset,
                _,
                _,
                agent_history_availability,
            ) = get_relative_poses(self.history_num_frames + 1, history_frames, track_id, history_agents,
                                   agent_from_world, agent_yaw_rad)

            current_other_actor = filter_agents_by_track_id(history_agents_flat, track_id)[0]
            current_type = np.argmax(current_other_actor["label_probabilities"])

            agent_history_feat = np.concatenate(
                [self.get_poly(agent_history_coords_offset[::-1][:]),
                 agent_history_timestamp, 
                 (self.max_num_agent - idx) * np.ones((self.history_num_frames, 1)), 
                 TYPE_INDEX[current_type] * np.ones((self.history_num_frames, 1))],
                axis=1
            )

            agent_history_availability = agent_history_availability[::-1]
            num_agents = idx + 1        

            all_other_agents_history_positions[idx, :, :] = agent_history_feat
            all_other_agents_history_availability[idx] = np.logical_and(
                agent_history_availability[:-1], agent_history_availability[1:],
            )

        other_agents_polyline = all_other_agents_history_positions.copy()
        other_agents_polyline_availability = all_other_agents_history_availability.copy()

        agent_dict = {
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
            "num_agents":num_agents,
        }

        return agent_dict

    def _vectorize_map(self, agent_centroid_m, agent_from_world):
        
        MAX_POINTS_LANES = self.cfg["data_generation_params"]["max_points_per_lane"]

        MAX_LANE_DISTANCE = self.cfg["data_generation_params"]["max_lane_distance"]
        INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN  # split lane polyline by fixed number of points
        STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane

        
        lanes_points = np.zeros((self.max_num_lanes * 2, MAX_POINTS_LANES-1, 7), dtype=np.float32)
        lanes_availabilities = np.zeros((self.max_num_lanes * 2, MAX_POINTS_LANES-1), dtype=np.float32)
        
        num_lane = 0
        # 8505 x 2 x 2
        lanes_bounds = self.mapAPI.bounds_info["lanes"]["bounds"]

        # filter first by bounds and then by distance, so that we always take the closest lanes
        lanes_indices = indices_in_bounds(agent_centroid_m, lanes_bounds, MAX_LANE_DISTANCE)
        distances = []
        for lane_idx in lanes_indices:
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
            lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - agent_centroid_m, axis=-1)
            min_dis = np.min(lane_dist)
            if min_dis <= MAX_LANE_DISTANCE:
                distances.append(min_dis)
        lanes_indices = lanes_indices[np.argsort(distances)]

        for out_idx, lane_idx in enumerate(lanes_indices):
            if out_idx == self.max_num_lanes:
                break

            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)

            xy_left = lane["xyz_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xyz_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = transform_points(xy_left, agent_from_world)
            xy_right = transform_points(xy_right, agent_from_world)

            num_vectors_left = len(xy_left)
            num_vectors_right = len(xy_right)

            lanes_left = np.zeros((MAX_POINTS_LANES, 2))
            lanes_right = np.zeros((MAX_POINTS_LANES, 2))
            
            lanes_left[:num_vectors_left, :] = xy_left
            lanes_right[:num_vectors_right, :] = xy_right

            lanes_availabilities[out_idx * 2, :num_vectors_left - 1] = 1
            lanes_availabilities[out_idx * 2 + 1, :num_vectors_right - 1] = 1

            num_lane = out_idx * 2 + 2

            lanes_points[out_idx * 2, :, :] = np.concatenate(
                [self.get_poly(lanes_left),#4
                 np.arange(MAX_POINTS_LANES-1).reshape(-1, 1),
                 (self.max_num_lanes * 2 - out_idx * 2 - 1) * np.ones((MAX_POINTS_LANES-1, 1)),
                 4 * np.ones((MAX_POINTS_LANES-1, 1))],  
                axis=1
            )

            lanes_points[out_idx * 2 + 1, :, :] = np.concatenate(
                [self.get_poly(lanes_right),#4
                 np.arange(MAX_POINTS_LANES-1).reshape(-1, 1),
                 (self.max_num_lanes * 2 - out_idx * 2)* np.ones((MAX_POINTS_LANES-1, 1)),
                 4 * np.ones((MAX_POINTS_LANES-1, 1))],  
                axis=1
            )

        return {
            "lanes": lanes_points,
            "lanes_availabilities": lanes_availabilities.astype(np.bool),
            "num_lane":num_lane,
        }
