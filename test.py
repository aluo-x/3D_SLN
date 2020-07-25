from options.options import Options
import os
import torch
import numpy as np
import sys

# print options to help debugging


if __name__ == '__main__':
    args = Options().parse()
    if (args.test_dir is not None) and (not os.path.isdir(args.test_dir)):
        os.mkdir(args.test_dir)
    if args.batch_gen:
        from testing.test_VAE import get_layouts_from_network

        get_layouts_from_network(args)
        exit()
    if args.measure_acc_l1_std:
        from testing.test_acc_mean_std import get_std, get_acc_l1

        get_acc_l1(args)
        get_std(args)
        exit()
    if args.heat_map:
        from testing.test_heatmap import produce_heatmap, plot_heatmap

        print("Calling network to produce object positions...")
        produce_heatmap(args)

        print("Rendering images...")
        test_data_dir = os.path.join(args.test_dir, "data")
        heat_dir = os.path.join(test_data_dir, "heat")
        room_idx=0
        heat_pkl_path=os.path.join(heat_dir, str(room_idx).zfill(4) + "_heat.pkl")
        save_path = heat_dir
        plot_heatmap(heat_pkl_path, save_path)
        exit()
    if args.draw_2d:
        from testing.test_plot2d import plot2d

        # Please follow this data format when calling the plot2d function
        # For rotation, you can use argmax or weighted average of network rotation prediction
        test_data_dir = os.path.join(args.test_dir, "data")
        save_2d = os.path.join(test_data_dir, "2D_rendered")
        exp_boxes = [[0.31150928139686584,0.3127100169658661,0.003096628002822399,0.7295752763748169,0.8262581825256348,0.054250866174697876],[-0.06599953025579453,0.017223943024873734,0.2885378897190094,0.2573782205581665,0.7553179860115051,0.42857787013053894],[0.5567594766616821,0.017786923795938492,0.142490953207016,0.9046159982681274,0.31667089462280273,0.6691973209381104],[0.6205720901489258,0.018211644142866135,0.8416993021965027,0.8348240852355957,0.3893248736858368,0.963701605796814],[0.171146959066391,0.017671708017587662,0.8085968494415283,0.4601595997810364,0.5026606321334839,0.9657217264175415],[0.0,0.0,0.0,1.0,0.7327236533164978,0.9278678297996521]]
        exp_boxes = [torch.from_numpy(np.array(x)).float() for x in exp_boxes]
        exp_rots = [0.0008550407364964485, 18.074506759643555, 6.062503337860107, 12.16077995300293, 12.012971878051758, 0.0]
        exp_rots = [torch.from_numpy(np.array(x)).float() for x in exp_rots]
        obj_types = [20, 18, 30, 3, 11, 0]
        # Last obj is the "room" bounding box
        # Last rotation doesn't matter (isn't used)
        plot2d(exp_boxes, exp_rots, obj_types, save_2d)
        exit()
    if args.draw_3d:
        from testing.test_plot3d import run_blender, run_blender_mask_depth

        # Note this requires running batch_gen first
        # Run the following to select which GPU Blender will use
        # And the path to the blender 2.79 binary
        # export CUDA_VISIBLE_DEVICES=1
        # export PATH="/data/vision/billf/mooncam/code/yonglong/blender:$PATH"
        blender_path = args.blender_path
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PATH"] += os.pathsep + blender_path
        run_blender(args)
        exit()
    if args.fine_tune:
        from testing.test_render_refine import finetune_VAE

        # Replace with list of room IDs
        room_to_finetune = ["7096"]
        base_save_dir = os.path.join(args.test_dir, "data", "finetune")
        if not os.path.isdir(base_save_dir):
            os.mkdir(base_save_dir)
        save_directories = [os.path.join(base_save_dir, x) for x in room_to_finetune]
        finetune_VAE(args, room_to_finetune, save_directories)
        exit()
    if args.gan_shade:
        # This loads weights, which is slow so we put it behind the flag
        from testing.test_SPADE_shade import colorize_with_spade
        # Modify render_semantic_depth to choose the room id
        # Quite slow, run on select rooms...
        blender_path = args.blender_path
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PATH"] += os.pathsep + blender_path
        run_blender_mask_depth(args)
        # Disable if the masks & depth already exist
        input_dir = os.path.join(args.test_dir, "data", "semantic_masks")
        output_dir = os.path.join(args.test_dir, "data", "SPADE_out")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Rooms can either be "all" or a list of strings
        colorize_with_spade(num_z=50, semantic_dir=input_dir, save_dir=output_dir, rooms="all")
        exit()
