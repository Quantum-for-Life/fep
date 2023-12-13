# WIP

This will likely only work with a CUDA compatible GPU. While OpenMM has support CPU, it chooses different code paths. See for example https://github.com/openmm/openmm/issues/3288 where the poster runs into different errors depending on whether the platform is CPU or GPU.

## Notes for myself
* [yank](https://github.com/choderalab/yank) does not linearly scale the reciprocal contribution from PME. Instead, they reweight the endstates by increasing the cutoff a lot. See [this function](https://github.com/choderalab/yank/blob/c06059045bcf86d610f2e39c6db3944994b9f392/Yank/yank.py#L1260) and the places it's called for info on how to use it. To a maximum of 16Å
