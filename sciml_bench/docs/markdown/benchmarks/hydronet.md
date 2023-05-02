# hydronet Benchmark 

The scientific value of SchNet model value is twofold: molecular property prediction and 
interatomic potential (when trained with forces). The water cluster data is unique in standard
benchmarks in that it provides long-range hydrogen bonding interactions. These hydrogen bonding
interactions are what makes water such a good solvent and imparts the unique properties of
water, so they are important to capture not only for modelling water itself but for modelling
molecules and biomolecules solvated in water. The output is the predicted property. For qm9,
this is “energy_U” (internal energy at 298.15 K). For the water dataset, this is “energy”
(binding energy).The Hydronet benchmark uses the Schnet model which calculates energy and
forces. There are about ~200k trainable parameters. The inference operation over the entire
test set be performed by running the "test_set_errors.py" program.

* Main Domain: Material Sciences
* Sub Domain: 
* Task: Estimation
* Relevant Datasets: hydronet_ds1
* Implementation: PyTorch
* Support for Distributed Training: No
* Device Support: CPU / GPU
* Authors: Sutanay Choudhury, Jenna Pope, Jesun Firoz, Hatem Helal

<!--
The scientific value of SchNet model value is twofold: molecular property prediction and 
interatomic potential (when trained with forces). The water cluster data is unique in standard
benchmarks in that it provides long-range hydrogen bonding interactions. These hydrogen bonding
interactions are what makes water such a good solvent and imparts the unique properties of
water, so they are important to capture not only for modelling water itself but for modelling
molecules and biomolecules solvated in water. The output is the predicted property. For qm9,
this is “energy_U” (internal energy at 298.15 K). For the water dataset, this is “energy”
(binding energy).The Hydronet benchmark uses the Schnet model which calculates energy and
forces. There are about ~200k trainable parameters. The inference operation over the entire
test set be performed by running the "test_set_errors.py" program.

* Main Domain: Material Sciences
* Sub Domain: 
* Task: Estimation
* Relevant Datasets: hydronet_ds1
* Implementation: PyTorch
* Support for Distributed Training: No
* Device Support: CPU / GPU
* Authors: Sutanay Choudhury, Jenna Pope, Jesun Firoz, Hatem Helal
-->
