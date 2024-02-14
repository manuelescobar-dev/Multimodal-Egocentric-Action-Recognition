# Structure

# Usage
## Arguments (args)
1. Loads default arguments from the `default.yaml` file in the `configs` folder.
2. In command line `--config`: Path to the config file, else debug.yaml is used.
3. Merges the arguments from the command line with the default arguments.
4. Merges path configurations with the ones in config file.
5. Adds log directories to args.