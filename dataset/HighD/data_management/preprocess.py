import numpy as np
import pandas as pd
from data_management.read_csv import *
from IPython import embed

PROCESSED_DATA_PATH = (
    "D:/Workspace/UncertaintySafeField/highD/data/highd_changeLane_egoAxis/"
)

TIME_FORWARD = 5
TIME_BACKWARD = 3
ALL_RELATIVE_ID = "allId"
OTHER_RELATIVE_ID = {
    PRECEDING_ID,
    FOLLOWING_ID,
    LEFT_PRECEDING_ID,
    LEFT_ALONGSIDE_ID,
    LEFT_FOLLOWING_ID,
    RIGHT_PRECEDING_ID,
    RIGHT_ALONGSIDE_ID,
    RIGHT_FOLLOWING_ID,
}  # 均以换道前相对位置为准


PRECEDING_X = "precedingX"
PRECEDING_Y = "precedingY"

FOLLOWING_X = "followingX"
FOLLOWING_Y = "followingY"
FOLLOWING_X_VELOCITY = "followingXVelocity"

LEFT_PRECEDING_X = "leftPrecedingX"
LEFT_PRECEDING_Y = "leftPrecedingY"
LEFT_PRECEDING_X_VELOCITY = "leftPrecedingXVelocity"

LEFT_ALONGSIDE_X = "leftAlongsideX"
LEFT_ALONGSIDE_Y = "leftAlongsideY"
LEFT_ALONGSIDE_X_VELOCITY = "leftAlongsideXVelocity"

LEFT_FOLLOWING_X = "leftFollowingX"
LEFT_FOLLOWING_Y = "leftFollowingY"
LEFT_FOLLOWING_X_VELOCITY = "leftFollowingXVelocity"

RIGHT_PRECEDING_X = "rightPrecedingX"
RIGHT_PRECEDING_Y = "rightPrecedingY"
RIGHT_PRECEDING_X_VELOCITY = "rightPrecedingXVelocity"

RIGHT_ALONGSIDE_X = "rightAlongsideX"
RIGHT_ALONGSIDE_Y = "rightAlongsideY"
RIGHT_ALONGSIDE_X_VELOCITY = "rightAlongsideXVelocity"

RIGHT_FOLLOWING_X = "rightFollowingX"
RIGHT_FOLLOWING_Y = "rightFollowingY"
RIGHT_FOLLOWING_X_VELOCITY = "rightFollowingXVelocity"

TARGET_PRECEDING_ID = "targetPrecedingId"
TARGET_PRECEDING_X = "targetPrecedingX"
TARGET_PRECEDING_Y = "targetPrecedingY"
TARGET_PRECEDING_X_VELOCITY = "targetPrecedingXVelocity"

TARGET_FOLLOWING_ID = "targetFollowingId"
TARGET_FOLLOWING_X = "targetFollowingX"
TARGET_FOLLOWING_Y = "targetFollowingY"
TARGET_FOLLOWING_X_VELOCITY = "targetFollowingXVelocity"

TARGET_VEH_ID = "targetVehId"
TARGET_SET_ID = "targetSetId"


class LaneChange:
    def __init__(
        self,
        targ_veh_id,
        targ_frame,
        rela_vehs,
        begin_frame,
        end_frame,
        read_tracks,
        static_info,
        meta_dictionary,
    ):
        self.tracks = read_tracks
        self.static_info = static_info
        self.meta_dictionary = meta_dictionary

        self.target_set_id = self.meta_dictionary[ID]
        self.targ_veh_id = targ_veh_id
        self.targ_frame = targ_frame
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.rela_vehs = rela_vehs

        total_frame_num = (TIME_FORWARD + TIME_BACKWARD) * self.meta_dictionary[
            FRAME_RATE
        ]
        self.df = pd.DataFrame(
            {
                X: np.zeros(total_frame_num),
                Y: np.zeros(total_frame_num),
                X_VELOCITY: np.zeros(total_frame_num),
                Y_VELOCITY: np.zeros(total_frame_num),
                PRECEDING_X: np.zeros(total_frame_num),
                PRECEDING_Y: np.zeros(total_frame_num),
                PRECEDING_X_VELOCITY: np.zeros(total_frame_num),
                FOLLOWING_X: np.zeros(total_frame_num),
                FOLLOWING_Y: np.zeros(total_frame_num),
                FOLLOWING_X_VELOCITY: np.zeros(total_frame_num),
                TARGET_PRECEDING_X: np.zeros(total_frame_num),
                TARGET_PRECEDING_Y: np.zeros(total_frame_num),
                TARGET_PRECEDING_X_VELOCITY: np.zeros(total_frame_num),
                TARGET_FOLLOWING_X: np.zeros(total_frame_num),
                TARGET_FOLLOWING_Y: np.zeros(total_frame_num),
                TARGET_FOLLOWING_X_VELOCITY: np.zeros(total_frame_num),
                LEFT_PRECEDING_X: np.zeros(total_frame_num),
                LEFT_PRECEDING_Y: np.zeros(total_frame_num),
                LEFT_PRECEDING_X_VELOCITY: np.zeros(total_frame_num),
                LEFT_ALONGSIDE_X: np.zeros(total_frame_num),
                LEFT_ALONGSIDE_Y: np.zeros(total_frame_num),
                LEFT_ALONGSIDE_X_VELOCITY: np.zeros(total_frame_num),
                LEFT_FOLLOWING_X: np.zeros(total_frame_num),
                LEFT_FOLLOWING_Y: np.zeros(total_frame_num),
                LEFT_FOLLOWING_X_VELOCITY: np.zeros(total_frame_num),
                RIGHT_PRECEDING_X: np.zeros(total_frame_num),
                RIGHT_PRECEDING_Y: np.zeros(total_frame_num),
                RIGHT_PRECEDING_X_VELOCITY: np.zeros(total_frame_num),
                RIGHT_ALONGSIDE_X: np.zeros(total_frame_num),
                RIGHT_ALONGSIDE_Y: np.zeros(total_frame_num),
                RIGHT_ALONGSIDE_X_VELOCITY: np.zeros(total_frame_num),
                RIGHT_FOLLOWING_X: np.zeros(total_frame_num),
                RIGHT_FOLLOWING_Y: np.zeros(total_frame_num),
                RIGHT_FOLLOWING_X_VELOCITY: np.zeros(total_frame_num),
            }
        )
        self.meta_df = pd.Series(rela_vehs)
        self.meta_df.drop(ALL_RELATIVE_ID, inplace=True)
        self.meta_df[TARGET_SET_ID] = self.meta_dictionary[ID]

    def update_rela_data(self, key, track):
        init_frame = track[FRAME][0]
        final_frame = track[FRAME][-1]
        begin_index = self.begin_frame - init_frame
        end_index = self.end_frame - init_frame
        zeros_before = np.zeros(0)
        zeros_after = np.zeros(0)
        if begin_index < 0:
            zeros_before = np.zeros(-begin_index)
            begin_index = 0
        if self.end_frame > final_frame:
            zeros_after = np.zeros(self.end_frame - final_frame)
            end_index -= self.end_frame - final_frame

        if key == ID:
            self.df[X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
            self.df[Y_VELOCITY] = np.concatenate(
                (zeros_before, track[Y_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif self.rela_vehs[key] == 0:
            pass
        elif key == PRECEDING_ID:
            self.df[PRECEDING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[PRECEDING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[PRECEDING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == FOLLOWING_ID:
            self.df[FOLLOWING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[FOLLOWING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[FOLLOWING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == LEFT_PRECEDING_ID:
            self.df[LEFT_PRECEDING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[LEFT_PRECEDING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[LEFT_PRECEDING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == LEFT_ALONGSIDE_ID:
            self.df[LEFT_ALONGSIDE_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[LEFT_ALONGSIDE_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[LEFT_ALONGSIDE_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == LEFT_FOLLOWING_ID:
            self.df[LEFT_FOLLOWING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[LEFT_FOLLOWING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[LEFT_FOLLOWING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == RIGHT_PRECEDING_ID:
            self.df[RIGHT_PRECEDING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[RIGHT_PRECEDING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[RIGHT_PRECEDING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == RIGHT_ALONGSIDE_ID:
            self.df[RIGHT_ALONGSIDE_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[RIGHT_ALONGSIDE_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[RIGHT_ALONGSIDE_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif key == RIGHT_FOLLOWING_ID:
            self.df[RIGHT_FOLLOWING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[RIGHT_FOLLOWING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[RIGHT_FOLLOWING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )

        if track[ID] != 0 and track[ID] == self.rela_vehs[TARGET_PRECEDING_ID]:
            self.df[TARGET_PRECEDING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[TARGET_PRECEDING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[TARGET_PRECEDING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )
        elif track[ID] != 0 and track[ID] == self.rela_vehs[TARGET_FOLLOWING_ID]:
            self.df[TARGET_FOLLOWING_X] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 0], zeros_after)
            )
            self.df[TARGET_FOLLOWING_Y] = np.concatenate(
                (zeros_before, track[BBOX][begin_index:end_index, 1], zeros_after)
            )
            self.df[TARGET_FOLLOWING_X_VELOCITY] = np.concatenate(
                (zeros_before, track[X_VELOCITY][begin_index:end_index], zeros_after)
            )

    def normalize(self):
        direct_positive = self.df[X_VELOCITY][0] > 0
        flipped = 1 if direct_positive else -1
        self.df *= flipped

        recent_x = self.df[X]
        recent_y = self.df[Y]
        init_x = self.df[X][0]
        init_y = self.df[Y][0]

        self.df[PRECEDING_X] -= recent_x
        self.df[PRECEDING_X][self.df[PRECEDING_X_VELOCITY] == 0] = 0
        self.df[PRECEDING_Y] -= recent_y
        self.df[PRECEDING_Y][self.df[PRECEDING_X_VELOCITY] == 0] = 0

        self.df[FOLLOWING_X] -= recent_x
        self.df[FOLLOWING_X][self.df[FOLLOWING_X_VELOCITY] == 0] = 0
        self.df[FOLLOWING_Y] -= recent_y
        self.df[FOLLOWING_Y][self.df[FOLLOWING_X_VELOCITY] == 0] = 0

        self.df[TARGET_PRECEDING_X] -= recent_x
        self.df[TARGET_PRECEDING_X][self.df[TARGET_PRECEDING_X_VELOCITY] == 0] = 0
        self.df[TARGET_PRECEDING_Y] -= recent_y
        self.df[TARGET_PRECEDING_Y][self.df[TARGET_PRECEDING_X_VELOCITY] == 0] = 0

        self.df[TARGET_FOLLOWING_X] -= recent_x
        self.df[TARGET_FOLLOWING_X][self.df[TARGET_FOLLOWING_X_VELOCITY] == 0] = 0
        self.df[TARGET_FOLLOWING_Y] -= recent_y
        self.df[TARGET_FOLLOWING_Y][self.df[TARGET_FOLLOWING_X_VELOCITY] == 0] = 0

        self.df[LEFT_PRECEDING_X] -= recent_x
        self.df[LEFT_PRECEDING_X][self.df[LEFT_PRECEDING_X_VELOCITY] == 0] = 0
        self.df[LEFT_PRECEDING_Y] -= recent_y
        self.df[LEFT_PRECEDING_Y][self.df[LEFT_PRECEDING_X_VELOCITY] == 0] = 0

        self.df[LEFT_ALONGSIDE_X] -= recent_x
        self.df[LEFT_ALONGSIDE_X][self.df[LEFT_ALONGSIDE_X_VELOCITY] == 0] = 0
        self.df[LEFT_ALONGSIDE_Y] -= recent_y
        self.df[LEFT_ALONGSIDE_Y][self.df[LEFT_ALONGSIDE_X_VELOCITY] == 0] = 0

        self.df[LEFT_FOLLOWING_X] -= recent_x
        self.df[LEFT_FOLLOWING_X][self.df[LEFT_FOLLOWING_X_VELOCITY] == 0] = 0
        self.df[LEFT_FOLLOWING_Y] -= recent_y
        self.df[LEFT_FOLLOWING_Y][self.df[LEFT_FOLLOWING_X_VELOCITY] == 0] = 0

        self.df[RIGHT_PRECEDING_X] -= recent_x
        self.df[RIGHT_PRECEDING_X][self.df[RIGHT_PRECEDING_X_VELOCITY] == 0] = 0
        self.df[RIGHT_PRECEDING_Y] -= recent_y
        self.df[RIGHT_PRECEDING_Y][self.df[RIGHT_PRECEDING_X_VELOCITY] == 0] = 0

        self.df[RIGHT_ALONGSIDE_X] -= recent_x
        self.df[RIGHT_ALONGSIDE_X][self.df[RIGHT_ALONGSIDE_X_VELOCITY] == 0] = 0
        self.df[RIGHT_ALONGSIDE_Y] -= recent_y
        self.df[RIGHT_ALONGSIDE_Y][self.df[RIGHT_ALONGSIDE_X_VELOCITY] == 0] = 0

        self.df[RIGHT_FOLLOWING_X] -= recent_x
        self.df[RIGHT_FOLLOWING_X][self.df[RIGHT_FOLLOWING_X_VELOCITY] == 0] = 0
        self.df[RIGHT_FOLLOWING_Y] -= recent_y
        self.df[RIGHT_FOLLOWING_Y][self.df[RIGHT_FOLLOWING_X_VELOCITY] == 0] = 0

        self.df[X] -= init_x
        self.df[Y] -= init_y

    def save_and_get_meta(self):
        for track in self.tracks:
            # Get the id of the current track
            if track[TRACK_ID] in self.rela_vehs[ALL_RELATIVE_ID]:
                for key, id in self.rela_vehs.items():
                    if key != ALL_RELATIVE_ID and id == track[TRACK_ID]:
                        self.update_rela_data(key, track)
        self.normalize()

        self.df.to_csv(
            PROCESSED_DATA_PATH
            + f"set{self.target_set_id:02d}_veh{self.targ_veh_id:04d}.csv",
            index=False,
        )
        target_df = self.df.loc[self.targ_frame - self.begin_frame]
        return pd.concat([self.meta_df, target_df], axis=0)


class DataPreprocess:
    def __init__(self, arguments, read_tracks, static_info, meta_dictionary):
        self.tracks = read_tracks
        self.static_info = static_info
        self.meta_dictionary = meta_dictionary
        self.frame_rate = self.meta_dictionary[FRAME_RATE]
        self.frame_num_for = self.frame_rate * TIME_FORWARD
        self.frame_num_back = self.frame_rate * TIME_BACKWARD

    def run_and_get_meta(self):
        meta = []
        for target_track in self.tracks:
            # Get the id of the current track
            targ_veh_id = target_track[TRACK_ID]
            if self.static_info[targ_veh_id][NUMBER_LANE_CHANGES] == 1:
                init_frame = self.static_info[targ_veh_id][INITIAL_FRAME]
                final_frame = self.static_info[targ_veh_id][FINAL_FRAME]
                init_lane = target_track[LANE_ID][0]
                final_lane = target_track[LANE_ID][-1]
                for index, frame_num in enumerate(target_track[FRAME]):
                    if target_track[LANE_ID][index] == final_lane:
                        targ_index = index - 1  # -1表明关注的是换道前的周车信息
                        targ_frame = frame_num - 1
                        break
                if (
                    targ_frame - init_frame >= self.frame_num_for
                    and final_frame - targ_frame >= self.frame_num_back
                ):
                    print(
                        f"Set ID:{self.meta_dictionary[ID]:02d},\tVeh ID:{targ_veh_id},\tFrame:{targ_frame-init_frame}\t{final_frame-targ_frame}"
                    )
                    begin_frame = targ_frame - self.frame_num_for
                    end_frame = targ_frame + self.frame_num_back
                    rela_vehs = dict()
                    rela_vehs[ALL_RELATIVE_ID] = set()
                    for relative_id in OTHER_RELATIVE_ID:
                        rela_vehs[relative_id] = target_track[relative_id][targ_index]
                        rela_vehs[ALL_RELATIVE_ID].add(rela_vehs[relative_id])
                    rela_vehs[ALL_RELATIVE_ID].add(targ_veh_id)
                    rela_vehs[ID] = targ_veh_id
                    rela_vehs[TARGET_PRECEDING_ID] = target_track[PRECEDING_ID][
                        targ_index + 1
                    ]
                    rela_vehs[TARGET_FOLLOWING_ID] = target_track[FOLLOWING_ID][
                        targ_index + 1
                    ]

                    lane_change = LaneChange(
                        targ_veh_id,
                        targ_frame,
                        rela_vehs,
                        begin_frame,
                        end_frame,
                        self.tracks,
                        self.static_info,
                        self.meta_dictionary,
                    )
                    meta.append(lane_change.save_and_get_meta())
        meta_df = pd.concat(meta, axis=1).T
        # meta_df.to_csv(
        #     PROCESSED_DATA_PATH + f"meta_set{self.meta_dictionary[ID]:02d}.csv",
        #     index=False,
        # )
        return meta_df

