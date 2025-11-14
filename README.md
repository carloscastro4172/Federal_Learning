## Simple FL Package

### ```examples``` directory
Sample ML Engine codes that integrate FL client side libraries.

### ```fl_main``` directory
Agent, aggregator, database codes together with supporting libraries.

### ```setups``` directory
Configulation files with JSON format and installation yaml files.


## Install
For all the environments of FL Server (Aggregator), FL Client (Agent), and Database Server, please create conda environment and activate it.

```sh
# macOS
conda env create -n federatedenv -f ./setups/federatedenv.yaml
# Linux
conda env create -n federatedenv -f ./setups/federatedenv_linux.yaml
```

[comment]: <> (### FL Client &#40;Agent&#41; )

[comment]: <> (```sh)

[comment]: <> (# macOS)

[comment]: <> (conda env create -n federatedenv -f ./setups/federatedenv.yaml)

[comment]: <> (# Linux)

[comment]: <> (conda env create -n federatedenv -f ./setups/federatedenv_linux.yaml)

[comment]: <> (```)


[comment]: <> (### Database Server)

[comment]: <> (```sh)

[comment]: <> (# macOS)

[comment]: <> (conda env create -n federatedenv -f ./setups/federatedenv.yaml)

[comment]: <> (# Linux)

[comment]: <> (conda env create -n federatedenv -f ./setups/federatedenv_linux.yaml)

[comment]: <> (```)

Be sure to do ```conda activate federatedenv``` when you run the codes.

Note: The environment has ```Python 3.7.4```. There is some known issues of ```ipfshttpclient``` with ```Python 3.7.2 and older```.


## Usage

### Running Database and Aggregator

Here is how to configure the FL server side modules of database and aggregator.
1. Edit the configuration files in the setups folder. The configuration details are explained [here](setups/).
2. Run the following 2 modules as separated processes in the order of ```pseudo_db``` -> ```server_th```.

```python
python -m fl_main.pseudodb.pseudo_db
python -m fl_main.aggregator.server_th
```

### [Minimal Example](examples/minimal)

This sample does not have actual training. This could be used as a template for user implementation of ML Engine.
#### Sample Execution
1. Edit the configuration files (config_agent.json) in the setups folder. The configuration details are explained [here](setups/).
2. Make sure the Database and Aggregator servers are running already. 
   Then, run the minimal example as follows.

```python
python -m examples.minimal.minimal_MLEngine
```
#### Simulation
FL systems can be run multiple agents for simulation within the same machine by specifying the port numbers for agents. 
##### Agent side
```python
python -m examples.minimal.minimal_MLEngine [simulation_flag] [gm_recv_port] [agent_name]
```

- ```simulation_flag```: 1 if it's simulation
- ```gm_recv_port```: Port number waiting for global models from the aggregator. This will be communicated to the aggregator via a participate message.
- ```agent_name```: Name of the local agent and directory name storing the ```state``` and model files. This needs to be unique for every agent.

For example:
```python
# First agent
python -m examples.minimal.minimal_MLEngine 1 50001 a1
# Second agent
python -m examples.minimal.minimal_MLEngine 1 50002 a2
```
