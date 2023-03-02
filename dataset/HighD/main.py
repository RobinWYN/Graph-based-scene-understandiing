import json
import os
import sys
import pickle
import argparse

from data_management.read_csv import *
from visualization.visualize_frame import VisualizationPlot
from data_management.preprocess import *
import pandas as pd

# data_path = "I:/课题组/丰田AI/泛化场景理解/UncertaintySafeField/highD/data/highd_intention/data/"
data_path = "I:/课题组/丰田AI/泛化场景理解/UncertaintySafeField/data/"
# data_path = "I:/课题组/丰田AI/泛化场景理解/UncertaintySafeField/highD/data/highd_changeLane/data/"


def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")

    # --- AutoPlay ---
    parser.add_argument(
        "--autoplay",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="True if you want to visualize the data.",
    )

    # --- Target Data Number ---
    parser.add_argument("--data_num", default="01", type=str, help="Target Data Number")

    # --- Settings ---
    parser.add_argument(
        "--visualize",
        default=True,  # TODO
        type=lambda x: (str(x).lower() == "true"),
        help="True if you want to visualize the data.",
    )

    # --- Visualization settings ---
    parser.add_argument(
        "--plotBoundingBoxes",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to plot the bounding boxes or not.",
    )
    parser.add_argument(
        "--plotDirectionTriangle",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to plot the direction triangle or not.",
    )
    parser.add_argument(
        "--plotTextAnnotation",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to plot the text annotation or not.",
    )
    parser.add_argument(
        "--plotTrackingLines",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to plot the tracking lines or not.",
    )
    parser.add_argument(
        "--plotClass",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to show the class in the text annotation.",
    )
    parser.add_argument(
        "--plotVelocity",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to show the class in the text annotation.",
    )
    parser.add_argument(
        "--plotIDs",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: decide whether to show the class in the text annotation.",
    )

    # --- I/O settings ---
    parser.add_argument(
        "--save_as_pickle",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Optional: you can save the tracks as pickle.",
    )
    parsed_arguments = vars(parser.parse_args())
    return parsed_arguments


if __name__ == "__main__":
    created_arguments = create_args()

    data_num = created_arguments["data_num"]
    # meta = []
    # for data_num in range(1, 61):
    # data_num = f"{data_num:02d}"
    created_arguments["input_path"] = data_path + data_num + "_tracks.csv"
    created_arguments["input_static_path"] = data_path + data_num + "_tracksMeta.csv"
    created_arguments["input_meta_path"] = data_path + data_num + "_recordingMeta.csv"
    created_arguments["pickle_path"] = data_path + data_num + ".pickle"
    created_arguments["background_image"] = data_path + data_num + "_highway.jpg"

    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
        with open(created_arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(created_arguments)
        print("Finished importing the pickle file.")

    if created_arguments["save_as_pickle"] and not os.path.exists(
        created_arguments["pickle_path"]
    ):
        print("Save tracks to pickle file.")
        with open(created_arguments["pickle_path"], "wb") as fp:
            pickle.dump(tracks, fp)

    # Read the static info
    try:
        static_info = read_static_info(created_arguments)
    except:
        print(
            "The static info file is either missing or contains incorrect characters."
        )
        sys.exit(1)

    # Read the video meta
    try:
        meta_dictionary = read_meta_info(created_arguments)
        print(
            "data Id:",
            meta_dictionary["id"],
            "\t\tlocation Id:",
            meta_dictionary["locationId"],
        )
        print(
            "duration Time:",
            meta_dictionary["duration"],
            "\tframe Rate:",
            meta_dictionary["frameRate"],
        )
        print("num Vehicles:", meta_dictionary["numVehicles"])
        print("speed Limit:", meta_dictionary["speedLimit"])
    except:
        print("The video meta file is either missing or contains incorrect characters.")
        sys.exit(1)

    if tracks is None:
        print("Please specify the path to the tracks csv/pickle file.")
        sys.exit(1)
    if static_info is None:
        print("Please specify the path to the static tracks csv file.")
        sys.exit(1)
    if meta_dictionary is None:
        print("Please specify the path to the video meta csv file.")
        sys.exit(1)

    if created_arguments["visualize"]:
        visualization_plot = VisualizationPlot(
            created_arguments, tracks, static_info, meta_dictionary
        )
        visualization_plot.show()
    else:
        data_preprocess = DataPreprocess(
            created_arguments, tracks, static_info, meta_dictionary
        )
    #         meta.append(data_preprocess.run_and_get_meta())
    # meta_df = pd.concat(meta, axis=0)
    # meta_df.to_csv(
    #     PROCESSED_DATA_PATH + f"meta.csv", index=False,
    # )
