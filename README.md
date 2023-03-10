# multi-tier
Simulation code for "Joint Caching and Forwarding in Data Intensive Networks with Hybrid Storage Systems"

Requirements:
- Python version >= 3.9.7
- Latest versions of the following packages: `numpy`, `scipy`, `simpy`, `networkx`, `tinydb`

Example command to run a simulation:
```
python3 ./src/sim_runner.py -n sample_test -t abilene -d sample_data -c -2 -u https://gist.github.com/fvmutlu/0542d6323967041613ec42bd6f15884d/raw/cd054ad05b55cbb1d95356782267e16bdf34f301/sample_config.json
```

This command will generate some output data in the `sample_data.json` file found in the `test_outputs/` folder, as well as a record of the test config with the name `sample_test` in the `test_configs.json` file found in the same folder, which will include the time at which the test was run, how much time elapsed for the test to finish, as well as the parameters of the test.
The last argument in the command (`-u`) is a URL to a test config json hosted as a gist file. This allows writing test configs outside the source code.

Further instructions on how to run the code for different experiments and how to interpret the output data will be added shortly.
