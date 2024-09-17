# rapid-updating-of-movement-by-evidence
Repository for the paper ["Rapid, systematic updating of movement by accumulated decision evidence"](https://www.biorxiv.org/content/10.1101/2023.11.09.566389v2).

### Abstract
Acting in the natural world requires not only deciding among multiple options but also converting decisions into motor commands. How the dynamics of decision formation influence the fine kinematics of response movement remains, however, poorly understood. Here we investigate how the accumulation of decision evidence shapes the response orienting trajectories in a task where freely-moving rats combine prior expectations and auditory information to select between two possible options. Response trajectories and their motor vigor are initially determined by the prior. Rats movements then incorporate sensory information as early as 60 ms after stimulus onset by accelerating or slowing depending on how much the stimulus supports their initial choice. When the stimulus evidence is in strong contradiction, rats change their mind and reverse their initial trajectory. Human subjects performing an equivalent task display a remarkably similar behavior. We encapsulate these results in a computational model that maps the decision variable onto the movement kinematics at discrete time points, capturing subjects’ choices, trajectories and changes of mind. Our results show that motor responses are not ballistic. Instead, they are systematically and rapidly updated, as they smoothly unfold over time, by the parallel dynamics of the underlying decision process.

### Notebooks
We set up two notebooks:
- Figures: in which you will be able to load the data, available [here](https://osf.io/794vk/), and reproduce the main figures from the manuscript. All details from data are in the link provided and in the notebooks.
- MNLE_fitting_pipeline: in which you will see and play with the fitting procedure, starting from training a Mixed Neural Likelihood Estimator (MNLE, [Boelts et al. 2022](https://elifesciences.org/articles/77220)), and finding the approximate Maximum Likelihood Estimate with Bayesian Adaptive Direct Search (BADS, [Acerbi and Ma, 2017](https://papers.nips.cc/paper_files/paper/2017/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html)).

### Requeriments
- ```sbi```: [Tejero-Cantero et al., (2020)](https://joss.theoj.org/papers/10.21105/joss.02505), in which MNLE is developed. [Github](https://github.com/sbi-dev/sbi).
- ```pybads```: [Singh et al., (2024)](https://joss.theoj.org/papers/10.21105/joss.05694), [Github](https://github.com/acerbilab/pybads).


### Citation
@Article{MolanoMazon2023,
  author    = {Molano-Mazón, Manuel and Garcia-Duran, Alexandre and Pastor-Ciurana, Jordi and Hernández-Navarro, Lluís and Bektic, Lejla and Lombardo, Debora and de la Rocha, Jaime and Hyafil, Alexandre},
  title     = {Rapid, systematic updating of movement by accumulated decision evidence},
  year      = {2023},
  month     = nov,
  doi       = {10.1101/2023.11.09.566389},
  publisher = {Cold Spring Harbor Laboratory},
}
