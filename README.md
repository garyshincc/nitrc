# nitrc

Repository for analysis of the [CMIs HBN Data](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/).


## Setup

1. Download the subject-by-subject EEG Data from CMI's database.

2. The list of `tar` files to be placed in a directory at root level called `raw_data`.

3. Untar the data using the script `untar.sh`

At this point, the data is prepared.

4. Prepare python environment using `make setup` and `make install`

## Running

Activate your virtual environment using `source source_env.sh`.

All research are prepared in `research/scripts/...`. Simply run using python.
