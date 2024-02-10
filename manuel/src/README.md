# Structure
- `train_classifier.py`: 
- `save_feat.py`:
- `colab_runner.ipynb`:
- `utils`
  - `args.py`: Defines the arguments for the training and testing scripts. Reads the arguments from the command line and from the config folder files.
  - `epic_record.py`:
  - `loaders.py`:
  - `logger.py`:
  - `transforms.py`:
  - `utils.py`:
  - `video_record.py`:
- `train_val`
- `tasks`
- `pretrained_i3d`
- `models`: Contains the model classes.
  - `I3D.py`: Contains the I3D model.
  - `VideoModel.py`: To be implemented.
  - `FinalClassifier.py`: To be implemented.
- `configs`
- `action_net`

# Usage
## Arguments (args)
1. Loads default arguments from the `default.yaml` file in the `configs` folder.
2. In command line `--config`: Path to the config file, else debug.yaml is used.
3. Merges the arguments from the command line with the default arguments.
4. Merges path configurations with the ones in config file.
5. Adds log directories to args.