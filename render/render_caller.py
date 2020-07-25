import json
import os
import sys
#TODO: Port the option to force object mesh names, so we can faithfully render the diff_render output

from pathlib import Path
cur_file_path = str(Path(__file__).absolute())
sys.path.append(os.path.dirname(cur_file_path))

from render_room_color import render_test

argv = sys.argv
argv = argv[argv.index("--") + 1:]


def load_json(json_file):
    with open(json_file, 'r') as f:
        var = json.load(f)
        return var


def render_test_3d(test_dir):
    data = load_json(os.path.join(test_dir, "data", "data_extracted.json"))
    output_folder = os.path.join(test_dir, "data", "rendered")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for room_id, room_data in data.items():

        # Code to render the GT
        objs = room_data["gt"]["objs"]
        # boxes_gt = room_data["gt"]["boxes"]
        # angles_gt = room_data["gt"]["angles"]
        # out_path = os.path.join(test_dir, room_id + "_gt_3d.png")
        # render_scene_retrieve(objs, boxes_gt, angles_gt, output_folder, name=room_id + "_gt_" + str(k).zfill(2) + "_3d.png")

        # Code to render the generated scenes
        for k in range(4):
            boxes = room_data[str(k)]["boxes"]
            angles = room_data[str(k)]["angles"]
            # Change n_samples for higher quality renders
            render_test(objs,boxes,angles,output_folder, name=room_id + "_pred_" + str(k).zfill(2) + "_3d.png")


if __name__ == '__main__':
    test_dir = argv[0]
    # sideview = bool(int(argv[1]))
    render_test_3d(test_dir)
    exit()
