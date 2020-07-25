# 3D SLN 
Code release for:

**End-to-End Optimization of Scene Layout**  
Andrew Luo, Zhoutong Zhang, Jiajun Wu, and Joshua B. Tenenbaum  
CVPR 2020 (Oral) [Project site](http://3dsln.csail.mit.edu/),  [Bibtex](http://3dsln.csail.mit.edu/bibtex/3dsln_cvpr.bib)  
For help contact `afluo [a.t] andrew.cmu.edu` or open an issue
* Requirements  
   * Pytorch 1.2 (for everything)
   * [Neural 3D Mesh Renderer - daniilidis version](https://github.com/daniilidis-group/neural_renderer)   (for scene refinement only)
   For numerical stability, please modify projection.py to remove the multiplication by 0. 
   After the change `L33, L34` looks like: 
   ```
   x__ = x_
   y__ = y_ 
   ```
   
   * Blender 2.79 (for 3D rendering of rooms only)
     * Please install numpy in Blender
   * matplotlib
   * numpy
   * skimage (for SPADE based shading)
   * imageio (for SPADE based shading)
   * shapely (eval only)
   * PyWavefront (for scene refinement only, loading of 3d meshes)
   * PyMesh (for scene refnement only, remeshing of SUNCG objects)
   * 1 Nvidia GPU
  
Download checkpoints [here](https://u.pcloud.link/publink/show?code=XZcDaNkZiaNSUcUK57R7aLJH9Pr3kyzWGqMk), download metadata [here](https://u.pcloud.link/publink/show?code=XZaDaNkZ4bapJodBqTjUBrpyMqUT8zqqCQHk)
```
Project structure
|-3d_SLN
  |-data
    |-suncg_dataset.py
      # Actual definition for the dataset object, makes batches of scene graphs
  |-metadata
    # SUNCG meta data goes here
    |-30_size_info_many.json
      # data about object size/volume, for 30/70 cutoff
    |-data_rot_train.json
      # Normalized object positions & rotations for training
    |-data_rot_val.json
      # For testing
    |-size_info_many.json
      # data about object size/volume, different cutoff
    |-valid_types.json
      # What object types we should use for making the scene graph
      # Caution when editing this, quite a bit is hard coded elsewhere
  |-models
    |-diff_render.py
      # Uses the Neural Mesh Renderer (Pytorch Version) to refine object positions
    |-graph.py
      # Graph network building blocks
    |-misc.py
      # Misc helper functions for the diff renderer
    |-Sg2ScVAE_model.py
      # Code to construct the VAE-graph network
    |-SPADE_related.py
      # Tools to construct SPADE VAE GAN (inference only)
  |-options
    # Global options
  |-render
    # Contains various "profiles" for Blender rendering
  |-testing
    # You must call batch_gen in test.py at least once
    # It will call into get_layouts_from_network in test_VAE.py
    # this will compute the posterior mean & std and cache it
    |-test_acc_mean_std.py
      # Contains helper functions to measure acc/l1/std 
    |-test_heatmap.py
      # Contains the functions *produce_heatmap* and *plot_heatmap*
      # The first function takes as input a verbally defined scene graph
        # If not provided, it uses a default scene graph with 5 objects
        # It will load weights for a VAE-graph network
        # Then load the computed posterior mean & std
        # And repeatedly sample from the given scene graph
        # Saves the results to a .pkl file
      # The second function will load a .pkl and plot them as heatmaps
    |-test_plot2d.py
      # Contains a function that uses matplotlib
      # Does NOT require SUNCG
      # Plots the objects using colors provided by ScanNet
    |-test_plot3d.py
      # Calls into the blender code in the ../render folder
      # Requires the SUNCG meshes
      # Requires Blender 2.79
      # Either uses the CPU (Blender renderer)
      # Or uses the GPU (Cycles renderer)
      # Loads a HDR texture (from HDRI Haven) for background
    |-test_SPADE_shade.py
      # Loads semantic maps & depth map, and produces RGB images using SPADE
    |-test_utils.py
      # Contains helper functions for testing
        # Of interest is the *get_sg_from_words* function
    |-test_VAE.py
  |-build_dataset_model.py
     # Constructs dataset & dataloader objects
     # Also constructs the VAE-graph network
  |-test.py
     # Provides functions which performs the following:
       # generation of layouts from scene graphs under the *batch_gen* argument
       # measure the accuracy of l1 loss, accuracy, std under the *measure_acc_l1_std* argument
       # draw the topdown heatmaps of layouts with a single scene graph under the *heat_map* argument
       # plot the topdown boxes of layouts with under the *draw_2d* argument
       # plot the viewer centric layouts using suncg meshes under the *draw_3d* argument
       # perform SPADE based shading of semantic+depth maps under the *gan_shade* argument
  |-train.py
     # Contains the training loop for the VAE-graph network
  |-utils.py
     # Contains various helper functions for:
       # managing network losses
       # make scene graphs from bounding boxes
       # load/write jsons
       # misc other stuff
```
* Training the VAE-graph network (limited to 1 GPU):  
`python train.py`

* Testing the VAE-graph network:  
First run `python test.py --batch_gen` at least once. This computes and caches a posterior for future sampling using the training set. It also generates a bunch of layouts using the test set.

* To generate a heatmap:  
`python test.py --heat_map`  
You can either define your own scene graph (see the `produce_heatmap` function in `testing/test_heatmap.py`), if you do not provide one it will use the default one. The function will convert scene graphs defined using words into a format usable by the network.

* To compute STD/L1/Acc:  
`python test.py --measure_acc_l1_std`

* To plot the scene from a top down view with ScanNet colors (doesn't requrie SUNCG):  
`python test.py --plot2d`  
Please provide a (O+1 x 6) tensor of bounding boxes, and a (O+1,) tensor of rotations. The last object should be the bounding box of the room

* To plot 3D  
`python test.py --plot3d`  
This calls into `test_plot3d.py`, which in turn launched Blender, and executes `render_caller.py`, you can put in specific rooms by editing this file. The full rendering function is located in `render_room_color.py`. 

* To use a neural renderer to refine a room  
`python test.py --fine_tune`
Please select the indexes of the room in `test.py`. This will call into `test_render_refine.py` which uses the differentiable renderer located in `diff_render.py`. Learning rate, and loss types/weightings can be set in `test_render_refine.py`.  
We set a manual seed for demonstration purposes, in practice please remove this.


* To use SPADE to generate texture/shading/lighting for a room from semantic + depth  
`python test.py --gan_shade`
This will first call into `semantic_depth_caller.py` to produce the semantic and depth maps, then use SPADE to generate RGB images.

