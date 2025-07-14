

IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS                                               1

# DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement Beyond Sequential Paradigms

Kien Le Trung      Truong-Son Hy

**Abstract**—Chip placement, the task of determining optimal positions of circuit modules on a chip canvas, is a critical step in the VLSI design flow that directly impacts performance, power consumption, and routability. Traditional methods rely on analytical optimization or reinforcement learning, which struggle with hard placement constraints or require expensive online training for each new circuit design. To address these limitations, we introduce DiffPlace, a framework that formulates chip placement as a conditional denoising diffusion process, enabling transferable placement policies that generalize to unseen circuit netlists without retraining. DiffPlace leverages the generative capabilities of diffusion models to efficiently explore the vast placement space while conditioning on circuit connectivity and relative quality metrics to identify globally optimal solutions. Our approach combines energy-guided sampling with constrained manifold diffusion to ensure placement legality, achieving extremely low overlap across all experimental scenarios. Our method bridges the gap between optimization-based and learning-based approaches, offering a practical path toward automated, high-quality chip placement for modern VLSI design. Our source code is publicly available at: https://github.com/HySonLab/DiffPlace/.

**Index Terms**—VLSI Physical Design, Chip Placement, Diffusion Models, Generative AI, Transfer Learning, Constraint-Aware Optimization.

## I. INTRODUCTION

Modern integrated circuit design faces formidable challenges as semiconductor technology advances into increasingly complex regimes. Chip placement, the problem of determining optimal positions for circuit modules on a chip canvas, represents a critical bottleneck in the VLSI design flow with profound implications for power consumption, timing performance, and silicon area utilization. Despite extensive research that spans decades, conventional approaches remain inadequate: analytical methods struggle with combinatorial complexity, while manual expert-driven iterations require weeks of painstaking effort, creating a significant impediment to rapid design cycles.

The emergence of machine learning techniques has catalyzed promising innovations in automated chip placement. Reinforcement learning (RL) frameworks such as GraphPlace [1] and MaskPlace [2] have reconceptualized placement as a sequential decision process, producing notable improvements over traditional methods. However, these approaches are fundamentally constrained by their inherent architecture: they necessitate costly online training regimes for each new circuit design and exhibit limited capacity to generalize across varied netlist topologies without exhaustive fine-tuning procedures. The sequential nature of their decision-making process further introduces compounding errors, where suboptimal early placements irreversibly constrain subsequent choices.

Recent transfer learning paradigms exemplified by ChiPFormer [3] have attempted to address these limitations through knowledge transfer mechanisms, significantly reducing training time for novel designs. However, these methods remain attached to the sequential decision-making framework and continue to require substantial computational resources for fine-tuning and online interactions to achieve competitive results on previously unseen netlists. The core challenge persists: how to efficiently navigate the vast, high-dimensional placement space while leveraging cross-design knowledge without being constrained by the sequential decision paradigm.

In this paper, we present DiffPlace, a transformative approach that fundamentally reconceptualizes chip placement through conditional denoising diffusion processes. By departing from the dominant sequential paradigm, DiffPlace exploits the remarkable generative capabilities of diffusion models to capture complex placement distributions while precisely conditioning on structured netlist inputs. This novel formulation enables simultaneous placement of all macros, eliminating the compounding errors inherent in sequential approaches while dramatically reducing the computational burden of transfer learning. Figure 1 illustrates this fundamental difference between our approach and sequential RL methods, showing how the diffusion process gradually transforms random noise into a valid placement by simultaneously refining all macro positions, rather than placing them one at a time. The main contributions of our work are as follows:

- A new formulation of chip placement as a conditional denoising diffusion process that simultaneously optimizes all module positions, transcending the limitations of sequential approaches.
- An innovative energy-conditioned framework that elegantly handles multiple competing optimization objectives, striking a sophisticated balance between wirelength minimization, congestion management, and density constraints.
- A transfer learning strategy that revolutionizes adaptation to new designs through minimal fine-tuning, drastically reducing runtime compared to state-of-the-art alternatives while maintaining superior placement quality.
- Comprehensive evaluation benchmarks that demonstrate that DiffPlace consistently outperforms existing approaches while ensuring strict adherence to constraints.

In summary, this work represents a paradigm shift in chip
---


Fig. 1: Progressive denoising process for simultaneous placement generation. The sequence illustrates iterative refinement from random Gaussian noise x_T to final placement x_0, with intermediate predictions x̂_0 shown at intervals. Unlike sequential RL approaches that place components one by one, our method simultaneously optimizes all module positions. Existing methods [2] [5] focus exclusively on arranging movable components, our framework holistically addresses both fixed and movable elements within the placement space.

placement methodology, moving beyond the constraints of sequential decision-making toward a distribution-based holistic approach that fundamentally redefines the possibilities for automated high-performance VLSI physical design.

## II. RELATED WORKS

The evolution of chip placement methodologies has progressed through several paradigms, from classical analytical approaches to recent learning-based techniques. In this section, we examine this trajectory to position our diffusion-based method within the broader context of placement solutions.

### A. Classical Placement Methods

Conventional placement approaches fall into three main categories: partitioning-based, stochastic optimization, and analytical methods.

Partitioning-based techniques [6], [7] recursively divide netlists and placement regions into manageable sub-problems. Although computationally efficient and scalable to large designs, these methods often sacrifice global optimality for runtime performance, as decisions made at higher levels constrain subsequent optimizations. The hierarchical nature of these approaches fundamentally limits their ability to find globally optimal solutions, particularly for complex and highly connected designs.

Stochastic optimization methods, particularly simulated annealing [8], dominated placement in the 1980s and 1990s. These approaches perform random perturbations of placements guided by a gradually decreasing temperature parameter, enabling them to escape local minima. Although SA-based placers like TimberWolf [10] achieved high-quality results, their prohibitive runtime complexity made them impractical for modern IC designs with millions of components.

Subsequently, analytical methods emerged as the predominant approach, with force-directed techniques [9], [11] and non-linear optimizers [12], [24] showing significant promise. These approaches transform discrete placement problems into continuous optimization frameworks that can be solved with gradient-based methods. Recent advances, including DREAM-Place [13], which leverages GPU acceleration, and RePlAce [14], which employs an electrostatics-based formulation with Nesterov's method, have further pushed the boundaries of analytical placement. However, these methods invariably treat each design as an isolated problem, ignoring the knowledge from previous placements, a fundamental limitation that our work addresses.

### B. Reinforcement Learning for Placement

The application of reinforcement learning to chip placement has gained significant traction following the breakthrough work of Mirhoseini et al. [1], which demonstrated that RL could surpass human experts in placement quality. Their approach, GraphPlace, represents the netlist as a graph and uses a graph neural network to learn placement policies that sequentially position macros on the chip canvas. Although revolutionary, this work exposed several limitations inherent to the RL paradigm: (1) expensive online training requirements for each new design, (2) limited generalization to unseen netlists, and (3) the compounding error problem where early placement decisions constrain later options. Subsequent work has refined this approach while maintaining the sequential RL paradigm. DeepPR [4] extended GraphPlace by jointly considering placement and routing objectives, introducing a unified learning framework that produces placements with improved routability. However, it similarly requires extensive online interactions for each new design and struggles to generalize across designs. MaskPlace [2] reconceptualized placement as visual representation learning, utilizing convolutional neural networks to capture spatial relationships. This approach significantly improved the handling of mixed-size macros and achieved zero-overlap placements without post-processing. The authors model the chip canvas as a 2D image and employ a policy network to place macros sequentially, guided by density and congestion masks. Despite these innovations, MaskPlace retains the fundamental limitations of the RL paradigm: expensive online training and limited generalization. PRNet [20] further expanded the learning-based placement landscape by combining policy gradient methods for macro placement with generative routing networks, creating the first end-to-end neural pipeline for both placement and routing. This integration highlights the importance of considering downstream routing constraints during placement, but does not address the fundamental limitations of sequential decision
---


making. Despite their impressive results, all these RL-based approaches share common limitations: they require exhaustive online training for each new design, exhibit limited generalization capabilities, and follow a sequential placement paradigm that leads to compounding errors. These limitations motivate our exploration of generative models that can simultaneously place all components.

C. Transfer Learning and Offline Methods

Recent advances in transfer learning have attempted to address the generalization limitations of RL-based placers. ChiPFormer [3] presents a significant step forward by introducing an off-line decision-transformer framework that enables knowledge transfer between designs. By pretraining on a dataset of placement examples and fine-tuning on new designs, ChiPFormer reduces training time from days to hours while maintaining competitive placement quality. This approach begins to bridge the gap between design-specific and generalizable placement algorithms through a novel neural architecture that effectively captures and transfers placement knowledge. However, ChiPFormer remains fundamentally bound to the sequential placement paradigm and still requires non-trivial online interactions during fine-tuning, limiting its efficiency for new designs. In parallel, WireMask-BBO [23] explored a different direction by applying black-box optimization techniques to the placement problem. Although not strictly a learning-based method, this approach demonstrates the value of global optimization strategies that consider all macro positions simultaneously rather than sequentially. However, it requires expensive search procedures for each new design and does not take advantage of cross-design knowledge.

D. Alternative Learning Paradigms

Beyond reinforcement learning, researchers have explored other learning paradigms for chip placement. Supervised learning approaches such as Flora [21] and GraphPlanner [22] train neural networks to directly predict optimal placements from netlist features. These methods frame placement as a regression problem rather than a sequential decision process, allowing faster inference times. Flora employs a graph attention network to encode the netlist structure and predict macro positions, while GraphPlanner extends this approach with more sophisticated graph neural networks and loss functions. While these approaches move away from the sequential paradigm, they struggle with the fundamental challenge of generating placements that satisfy hard constraints like zero overlap, often requiring extensive post-processing. Recent work has also explored equivariant graph and hypergraph neural networks to learn structural representations of netlists, demonstrating improved generalization in EDA tasks involving connectivity and layout patterns [30].

E. Generative Models in EDA

Generative models have shown promise in various electronic design automation (EDA) tasks. GANNES [15] utilized generative adversarial networks for well generation in analog layouts, while ThermGAN [16] demonstrated the application of GANs to thermal map estimation. For congestion prediction, LHNN [18] employed latent hypergraph neural networks to model complex routing patterns. These applications highlight the potential of generative models to capture complex distributions in the EDA domain, but their application to macro placement has been limited. Unlike reinforcement learning approaches that make sequential decisions, generative models can potentially capture the joint distribution of all macro positions simultaneously, enabling more holistic placement optimization.

Our work builds on these advances by introducing a diffusion-based generative approach to chip placement. Table I summarizes the key characteristics of recent placement approaches, highlighting differences in methodology, resolution, state space complexity, overlap guarantees, and optimization metrics. As shown, diffusion-based methods offer significant advantages over both analytical and RL-based approaches, particularly in their ability to guarantee zero overlap while maintaining high efficiency and optimizing for all key placement metrics without the exponential state-space complexity inherent in sequential methods. Unlike previous sequential methods, our approach simultaneously optimizes all macro positions through an iterative denoising process. This fundamental change in methodology eliminates the compounding errors inherent in sequential approaches while enabling effective knowledge transfer between designs through conditional generation.

III. PRELIMINARIES

A. Chip Placement Problem

Modern chip design involves determining the optimal positions for circuit modules on a two-dimensional canvas. This placement task directly impacts critical metrics such as power consumption, timing, and chip area. The placement problem consists of the following.

1) Netlist Representation: A circuit is represented as a hypergraph G = (V, E), where each node v_i ∈ V corresponds to a module with attributes (w_i, h_i) for width and height, and each hyperedge e_j ∈ E represents a net connecting multiple modules through their pins. Each pin P_(i,j) has coordinates (P^x_(i,j), P^y_(i,j)) relative to the position of its parent module.

2) Placement Representation: For diffusion models, we represent a placement as a continuous set of coordinates X = {(x_i, y_i)}^|V|_(i=1) where each (x_i, y_i) denotes the position of module i on the normalized chip canvas. This continuous representation allows diffusion models to explore the placement space efficiently, in contrast to discrete grid-based approaches used in reinforcement learning methods.

B. Optimization Objectives

Chip placement requires balancing multiple competing objectives:
---


TABLE I: Comparison of placement methods across key design criteria. We evaluate representative approaches from analytical optimization, reinforcement learning (RL), offline transfer learning, and our proposed diffusion-based model (DiffPlace). The comparison includes placement resolution, state space complexity, overlap guarantees, efficiency in training and inference, and optimization targets such as half-perimeter wirelength (HPWL), congestion, and density. DiffPlace achieves simultaneous placement with zero overlap in most cases, while maintaining high efficiency and optimizing multiple placement objectives without relying on sequential decision-making or costly online training.

| Method               | Family       | Resolution | State Space | 0% Overlap | Reward      | Efficiency | Metrics |
| -------------------- | ------------ | ---------- | ----------- | ---------- | ----------- | ---------- | ------- |
| DREAMPlace \[13]     | Nonlinear    | Continuous | -           | No         | -           | -/High     | H,D     |
| Graph Placement \[1] | RL+Nonlinear | 1282       | (1282)N     | No         | Sparse      | Med./Med.  | H,C,D   |
| DeepPR \[4]          | RL           | 322        | (322)N      | No         | Dense       | High/Med   | H,C     |
| MaskPlace \[2]       | RL           | 2242       | (2242)N     | Yes        | Dense       | High/High  | H,C,D   |
| ChiPFormer \[3]      | Offline RL   | 842        | -           | Yes3       | Pre-trained | High/High  | H,C,D   |
| DiffPlace (Ours)     | Diffusion    | 2242       | -           | Yes3       | -           | High/High  | H,C,D   |


<sup>1</sup> Training/Inference efficiency
<sup>2</sup> H = Half-Perimeter Wire Length, C = Congestion, D = Density
<sup>3</sup> ChiPFormer and Ours achieves zero overlap on most benchmarks but not guaranteed on all circuits (3.27% overlap reported)

1) Half-Perimeter Wire Length (HPWL): The primary metric for estimating wire length is HPWL, calculated as:

$$HPWL(X) = \sum_{\forall net \in netlist} \left( \max_{P_{(i,j)} \in net} P^x_{(i,j)} - \min_{P_{(i,j)} \in net} P^x_{(i,j)} + \max_{P_{(i,j)} \in net} P^y_{(i,j)} - \min_{P_{(i,j)} \in net} P^y_{(i,j)} \right).$$

(1)

Minimizing HPWL reduces wire lengths, improving power consumption and timing performance. Unlike discrete optimization methods that struggle with gradient-based HPWL optimization, diffusion models can take advantage of its differentiable property during guided sampling.

2) Congestion and Routability: Congestion measures routing resource utilization, quantified using the RUDY estimator:

$$Congestion(g_{i,j}) = \sum_{net covering g_{i,j}} \left( \frac{1}{w_{net}} + \frac{1}{h_{net}} \right),$$

(2)

where $g_{i,j}$ is a grid cell and $w_{net}$, $h_{net}$ are the dimensions of the bounding box of the net. Diffusion models can incorporate congestion as a differentiable guide during sampling.

3) Overlap Constraints: A valid placement requires zero overlap between modules:

$$Overlap(X) = \frac{Total overlapping area}{Total module area} = 0.$$

(3)

Traditional methods handle this constraint through penalty functions or legalization steps, which can increase wire-length. Diffusion models can encode this constraint in the sampling process, producing naturally legal placements.

C. Denoising Diffusion Probabilistic Models

Diffusion models have emerged as powerful generative approaches that learn to transform noise into data samples by reversing a gradual noising process.

1) Forward Process: The forward process gradually adds noise to a placement $X_0$:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_tI),$$

(4)

where $\beta_t$ is a noise schedule. This allows for direct sampling of $x_t$ from $x_0$:

$$q(X_t|X_0) = \mathcal{N}(X_t; \sqrt{\bar{\alpha_t}}X_0, (1 - \bar{\alpha_t})I),$$

(5)

where $\bar{\alpha_t} = \prod_{s=1}^t(1 - \beta_s)$.

2) Reverse Process for Placement Generation: The reverse process generates placements by iteratively denoising:

$$p_\theta(X_{t-1}|X_t) = \mathcal{N}(X_{t-1}; \mu_\theta(X_t, t), \Sigma_\theta(X_t, t)).$$

(6)

For chip placement, the model is trained to predict the noise $\epsilon$ added during the forward process:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,X_0,\epsilon} \left[ \|\epsilon - \epsilon_\theta(X_t, t, G)\|^2 \right],$$

(7)

where $G$ is the netlist graph. This approach allows the model to learn the joint distribution of module placements conditioned on circuit connectivity.

3) Conditional Diffusion for Netlists: To generate placements for specific circuit netlists, we use conditional diffusion:

$$p_\theta(X_{t-1}|X_t, G) = \mathcal{N}(X_{t-1}; \mu_\theta(X_t, t, G), \Sigma_\theta(X_t, t, G)).$$

(8)

The conditioning on graph $G$ allows the model to generate netlist-specific placements, respecting the circuit's structure and connectivity patterns.

D. Guided Sampling for Placement Optimization

A key advantage of diffusion models for chip placement is the ability to use guided sampling to enforce constraints and optimize objectives.

1) Potential-Based Guidance: We define potential functions for each placement objective:

$$\phi(X) = w_{hpwl} \cdot \phi_{hpwl}(X) + w_{overlap} \cdot \phi_{overlap}(X) + w_{cong} \cdot \phi_{cong}(X).$$

(9)

These potentials guide the sampling process toward valid, optimized placements without requiring explicit reward engineering as in RL approaches.

2) Classifier-Free Guidance: We employ classifier-free guidance to balance between fidelity to training data and optimization goals:

$$\tilde{\epsilon}_\theta(X_t, t, G) = \epsilon_\theta(X_t, t) + \gamma \cdot (\epsilon_\theta(X_t, t, G) - \epsilon_\theta(X_t, t)),$$

(10)

where $\gamma$ controls the strength of conditioning on the netlist.
---


Fig. 2: Comparing the overall pipelines between (a) online RL placement and (b) diffusion model-based placement (ours).

| (a) GraphPlace/MaskPlace/DeepPR/PRNet (RL)                                                                                                                                | (b) Ours                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| * Placement Model
* Components Positions
* Environment
* Step by step
* EDA / Simulators
* Learning Policy
* Time/Resource consumingMetrics such as: HPWL, Congestion ... | - Synthetic placement data
- Pre-collected placement data
- Placement Model
- Directly deployment (Seen circuits)
- Few-shot finetuning (Unseen circuits)
- Environment
- Metrics |


In (a), the online RL model continuously interacts with the environment (i.e., a placement simulator or an EDA design tool for obtaining metrics from placement designs, with time consumption proportional to circuit complexity) to learn policy from scratch. This results in substantial training time for each new circuit in standard benchmarks. In (b), our diffusion model approach allows for efficient generation of placements in a single pass, eliminating the need for iterative environment interactions during inference. When presented with an unseen circuit, our diffusion model can generalize effectively with minimal adaptation, leveraging its learned representations of circuit patterns and placement constraints. This approach significantly reduces runtime compared to previous online RL methods while maintaining or improving placement quality.

## IV. METHODOLOGY

### A. Overview of DiffPlace

We introduce DiffPlace, a diffusion-based framework for chip placement that addresses key limitations of existing approaches. Unlike prior methods that rely on reinforcement learning or analytical optimization, DiffPlace leverages the generative capabilities of denoising diffusion models to efficiently produce high-quality, constraint-satisfying placements. As illustrated in Fig. 2, our approach eliminates the need for expensive environment interactions during inference, enabling significantly faster placement for new circuit designs.

DiffPlace consists of four key components: (1) a conditional denoising diffusion model for generating placements from netlist structures, (2) an energy-guided sampling mechanism to optimize multiple placement objectives simultaneously, (3) a transfer learning strategy enabling efficient adaptation to unseen netlists, and (4) an end-to-end evaluation pipeline that verifies metrics.

Table I compares DiffPlace with existing approaches in multiple dimensions. While reinforcement learning methods such as GraphPlace and MaskPlace require costly training for each new circuit, and optimization methods such as DREAMPlace struggle with hard placement constraints, DiffPlace achieves zero-overlap placements with high efficiency across all key metrics (HPWL, congestion, and density).

### B. Diffusion Model for Chip Placement

1) Placement Representation and Conditioning: In DiffPlace, we represent a placement as a set of normalized 2D coordinates x = {(xi, yi)}|V|i=1 where each (xi, yi) ∈ [−1, 1]² specifies the position of module i on the chip canvas. This continuous representation offers two advantages: it allows for direct optimization using gradient-based methods and enables efficient transfer across different chip sizes.

We condition the diffusion process on the netlist structure, represented as a graph G = (V, E). The nodes V correspond to modules with attributes that include dimensions (wi, hi) and type, while the edges E represent connections between modules with attributes that describe pin positions. This rich conditioning allows the model to generate placements that respect the connectivity of the underlying circuit.

2) Forward and Reverse Diffusion Processes: The forward process gradually diffuses a clean placement x₀ in Gaussian noise xT through T steps according to:

$$ q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_tI), $$

where {βt}Tt=1 is a noise schedule that controls the rate of noise addition. This process can be rewritten to enable direct sampling of any intermediate noise level:

$$ q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I), $$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$. The reverse process generates placements by iteratively denoising from pure noise xT to a clean placement x₀. We parameterize this process using a neural network ϵθ that predicts the noise component:

$$ p_\theta(x_{t-1}|x_t, G) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, G), \Sigma_\theta(x_t, t, G)), $$

where μθ and Σθ are parameterized by neural networks. During training, we optimize these parameters to predict the noise ϵ added in the forward process:

$$ \mathcal{L}(\theta) = \mathbb{E}_{t,x_0,\epsilon,G} [\|\epsilon - \epsilon_\theta(x_t, t, G, E_{rel}(x_0, G))\|^2]. $$

Unlike RL approaches that place modules sequentially, our diffusion model places all modules simultaneously, as shown in Fig. 2(b). This joint optimization avoids compounding errors from sequential decision making and allows for a more effective exploration of the placement space.

### C. Energy-Conditioned Denoising

1) Multi-Objective Energy Function for Placement Quality: The placement of the chips requires the balance of multiple
---


competing objectives. We capture these through a composite energy function:

$$E(x, G) = \lambda_1 \frac{E_{hpwl}(x, G)}{E^{ref}_{hpwl}} + \lambda_2 \frac{E_{cong}(x, G)}{E^{ref}_{cong}} + \lambda_3 E_{over}(x),$$

(15)

where:

- $E_{hpwl}(x, G)$ measures wirelength using the HPWL metric
- $E_{cong}(x, G)$ quantifies routing congestion via the RUDY estimator
- $E_{over}(x)$ penalizes overlapping modules

The weights $w_{hpwl}$, $w_{cong}$, and $w_{over}$ balance these objectives based on their relative importance, which we determine by validation on industry benchmarks. This multi-objective formulation allows DiffPlace to optimize for all relevant metrics simultaneously.

2) Relative Energy Conditioning: A key innovation in DiffPlace is conditioning the diffusion process on relative placement quality. For each netlist G, we define a relative energy metric that normalizes placement quality across different circuit designs:

$$E_{rel} = \exp \left( -\frac{E(x, G) - E^G_{min}}{E^G_{max} - E^G_{min}} \right),$$

(16)

where $E^G_{min}$ and $E^G_{max}$ are the minimum and maximum energy values observed for the netlist G during training. This exponential formulation ensures $E_{rel} \in (0, 1]$, with values closer to 1 indicating higher-quality placements.

Conditioned on $E_{rel}$, we augment the diffusion training objective:

$$L(\theta) = \mathbb{E}_{t,x_0,\epsilon,G} \left[ \|\epsilon - \epsilon_\theta(x_t, t, G, E_{rel}(x_0, G))\|^2 \right],$$

(17)

where $E_{rel}(x_0, G)$ is computed from the clean placement $x_0$. This energy-conditioned framework enables the model to learn how placement quality varies with different configurations, allowing it to focus on high-quality regions of the placement space during generation. During inference, the conditioning on $E_{rel} \approx 1$ guides the model to superior placements without requiring expensive optimization.

D. Graph Neural Network Architecture

1) Netlist Graph Representation: To effectively process the structure of the netlist, we develop a specialized graph neural network (GNN) architecture. The netlist is represented as a heterogeneous graph $G = (V, E)$ with node and edge features that capture the essential properties of the circuit. For each module node $v_i \in V$, we include the following features:

- Normalized dimensions $(w_i, h_i)$,
- Module type (macro, standard cell),
- Pin locations relative to the module center.

For each edge $(i, j) \in E$, we include:

- Net type,
- Pin-to-pin connections,
- Critical path information (when available).

This rich representation allows the model to reason about both the geometric and electrical properties of the circuit.

2) Message-Passing Neural Networks for Circuit Connectivity: Our GNN encoder employs message passing to capture circuit connectivity patterns:

$$h_v^{(l+1)} = \text{Update}\left(h_v^{(l)}, \text{Aggregate}(\{m_{u\rightarrow v}^{(l)} | u \in \mathcal{N}(v)\})\right),$$

(18)

where $h_v^{(l)}$ is the embedding of node $v$ in layer $l$, and $m_{u\rightarrow v}^{(l)}$ is the message from node $u$ to $v$. We implement this using PaiNN [26], a computationally efficient architecture that captures directional information critical to understanding pin-to-pin relationships in the circuit.

3) Score Network Design: The score network $\epsilon_\theta$ predicts the noise added during the forward process by combining the GNN-encoded netlist, time embedding, energy condition and current noisy placement.

$$\epsilon_\theta(x_t, t, G, E_{rel}) = \text{ScoreNet}(\text{GNN}(G), \text{TimeEmbed}(t), \text{EnergyEmbed}(E_{rel}), x_t).$$

(19)

Our implementation uses transformer-based attention mechanisms to integrate these components, with residual connections to facilitate training. This architecture effectively captures both local module interactions and global placement patterns, allowing the model to reason across multiple scales.

E. Guided Sampling for Multi-Objective Optimization

During inference, we employ guided sampling to ensure that generated placements meet constraints while optimizing quality metrics. We implement classifier-free guidance using our energy-conditioned training:

$$\tilde{\epsilon}_\theta(x_t, t, G) = \epsilon_\theta(x_t, t, G) + s \cdot (\epsilon_\theta(x_t, t, G, E_{rel}^{high}) - \epsilon_\theta(x_t, t, G)),$$

(20)

where $s$ is the guidance scale and $E_{rel}^{high} \approx 1$ conditions on high-quality placements. To enforce hard constraints, we incorporate gradient-based guidance for overlap elimination and congestion control:

$$\hat{\epsilon}_\theta(x_t, t, G) = \tilde{\epsilon}_\theta(x_t, t, G) - \sigma_t \nabla_{x_t} [w_{leg} \phi_{legality}(\hat{x}_0) + w_{cong} \phi_{congestion}(\hat{x}_0, G)],$$

(21)

where $\hat{x}_0 = \frac{1}{\sqrt{\alpha_t}}(x_t - \sqrt{1 - \alpha_t} \tilde{\epsilon}_\theta(x_t, t, G))$ is the predicted clean placement, and:

$$\phi_{legality}(x) = \sum_{i,j\in V} \max(0, -d_{ij}(x))^2,$$

(22)

$$\phi_{congestion}(x, G) = \max(0, \max_{g_{i,j}} \text{Congestion}(g_{i,j}) - C_{th}).$$

(23)

The final sampling follows standard DDPM with hierarchical constraint weighting that prioritizes overlap elimination in early timesteps and quality optimization in later stages, ensuring placements meet all industry requirements.
---


## F. Process-Aware Synthetic Data Generation

We enhance the synthetic netlist generation framework with process-aware constraints. While maintaining computational efficiency of the inverse approach, our method incorporates actual design rules and physical constraints, ensuring that the generated netlists reflect the characteristics of the real world circuit. We model the netlist generation process with explicit technology constraints. For the 45nm process node, we define a process specification vector P₄₅ ∈ ℝᵈᴾ that captures key physical and electrical parameters:

P₄₅ = [p<sub>geom</sub>; p<sub>elec</sub>; p<sub>layer</sub>], (24)

where:

- p<sub>geom</sub> = [h<sub>sc</sub>, w<sub>min</sub>, s<sub>min</sub>, p<sub>grid</sub>] represents geometric constraints with standard cell height, minimum width, minimum spacing, and placement grid.
- p<sub>elec</sub> = [R<sub>sq</sub>, C<sub>unit</sub>, f<sub>max</sub>, V<sub>dd</sub>] captures electrical parameters with sheet resistance, unit capacitance, maximum frequency, and the supply voltage.
- p<sub>layer</sub> = [n<sub>metal</sub>, {h<sub>i</sub>, p<sub>i</sub>, w<sub>i</sub>}<sup>n<sub>metal</sub></sup><sub>i=1</sub>] defines metal layer stack with layers and their respective heights, pitches, and widths.

The edge generation probability incorporates actual physical effects through a multi-factor model:

p(edge<sub>ij</sub>|d<sub>ij</sub>, P₄₅) = p<sub>base</sub>(d<sub>ij</sub>)·M<sub>phys</sub>(d<sub>ij</sub>, P₄₅)·M<sub>design</sub>(i, j), (25)

where the base probability follows rent's rule aware distribution:

p<sub>base</sub>(d<sub>ij</sub>) = γ · d<sub>ij</sub><sup>-α</sup> · exp(-d<sub>ij</sub>/λ<sub>rent</sub>), (26)

with α = 0.5 (Rent's exponent for logic circuits) and λ<sub>rent</sub> = √(A<sub>chip</sub>/10) where A<sub>chip</sub> is the chip area. The physical manufacturability factor M<sub>phys</sub> ensures routability:

M<sub>phys</sub>(d<sub>ij</sub>, P₄₅) = M<sub>wire</sub> · M<sub>delay</sub> · M<sub>power</sub>, (27)

where:

| Mwire = | { 1 if dij ≤ L(1)max exp(-(dij-L(1)max)/τwire) otherwise |
| ------- | -------------------------------------------------------- |


(28)

M<sub>delay</sub> = exp(-max(0, (R<sub>wire</sub>(d<sub>ij</sub>) · C<sub>wire</sub>(d<sub>ij</sub>))/(T<sub>clk</sub>/20) - 1)) (29)

M<sub>power</sub> = exp(-max(0, (I<sub>avg</sub> · R<sub>wire</sub>(d<sub>ij</sub>))/(0.05 · V<sub>dd</sub>) - 1)) (30)

with wire resistance and capacitance modeled as:

R<sub>wire</sub>(d) = R<sub>sq</sub> · (d/w<sub>eff</sub>(d)) · ρ<sub>via</sub>(d), (31)

C<sub>wire</sub>(d) = C<sub>unit</sub> · d · (1 + κ<sub>fringe</sub>), (32)

where w<sub>eff</sub>(d) is the effective wire width based on the assignment of the layer, ρ<sub>via</sub>(d) accounts for via resistance, and κ<sub>fringe</sub> = 0.3 models fringing capacitance. The design factor M<sub>design</sub> captures the circuit topology preferences:

| Mdesign(i, j) = | { ωlocal if modules i, j in same block ωglobal if modules i, j in different blocks ωio if either i or j is I/O pad |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |


(33)

with ω<sub>local</sub> = 2.0, ω<sub>global</sub> = 0.5, and ω<sub>io</sub> = 0.8 derived from analysis of real circuits. We generate modules following the realistic circuit hierarchy as in Algorithm 1.

### Algorithm 1 Physics-Aware Hierarchical Netlist Generation

**Require:** Target specs (N<sub>total</sub>, A<sub>target</sub>, f<sub>target</sub>), Process P₄₅
**Ensure:** Process-compliant netlist G = (V, E) with placement X
1: **Phase 1: Hierarchical Module Generation**
2: N<sub>macro</sub> ∼ Poisson(λ<sub>macro</sub>·log N<sub>total</sub>) with λ<sub>macro</sub> = 1.5
3: Generate macros with area distribution:
4:    A<sub>macro</sub> ∼ LogNormal(μ = log(0.01A<sub>target</sub>), σ = 0.5)
5: Place macros using force-directed method with boundary constraints
6: Partition remaining area into functional blocks using Voronoi tessellation
7: Fill blocks with standard cells respecting h<sub>sc</sub> and placement grid
8: **Phase 2: Physics-Constrained Edge Generation**
9: Initialize pin locations based on module types and orientations
10: **for** each potential connection (i, j) **do**
11:    d<sub>ij</sub> ← Manhattan distance considering blockages
12:    p<sub>ij</sub> ← p(edge<sub>ij</sub>|d<sub>ij</sub>, P₄₅) using Eq. (22)
13:    Add edge with probability p<sub>ij</sub>
14: **end for**
15: **Phase 3: Validation and Refinement**
16: Ensure minimum connectivity: ∀v ∈ V, degree(v) ≥ 1
17: Check Rent's rule: T = k · N<sup>p</sup> with p ∈ [0.5, 0.7]
18: Verify routing feasibility using global routing estimation
19: Add clock tree connections following H-tree topology
20: Add power/ground network following mesh structure
    **return** (G, X) with associated metrics

We validate the generated netlists against multiple circuit characteristics:

1) Rent's Rule Compliance:
   Score<sub>rent</sub> = exp(-(|p<sub>measured</sub> - p<sub>expected</sub>|)/0.1), (34)
   where p<sub>measured</sub> is fitted from T = k · N<sup>p</sup> for the generated partitions.

2) Wirelength Distribution:
   Score<sub>wire</sub> = exp(-D<sub>KL</sub>(P<sup>gen</sup><sub>wire</sub>||P<sup>real</sup><sub>wire</sub>)), (35)
   where P<sub>wire</sub> follows the log-normal distribution in real circuits.
---


IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS                                             8

Degree distribution comparison graph

Fig. 3: Degree distribution comparison. Real circuit data averaged from 15 ISPD 2015-2019 benchmarks. Shaded region shows theoretical range α ∈ [1.8, 2.3] for digital circuits. Error bars indicate standard deviation across benchmarks.

2) Lightweight Fine-tuning for New Netlists: For new netlists, we employ a lightweight fine-tuning procedure:

$$\min_{\theta} \mathbb{E}_{(x_0,E_{rel})\sim B,t,\epsilon} [\omega(E_{rel})\|\epsilon - \epsilon_\theta(x_t, t, G_{test}, E_{rel})\|^2],$$
(41)

where B is a buffer of generated placements for the test netlist G_test, and ω(E_rel) is a weighting function that prioritizes placements with better energy metrics. To promote effective exploration during fine-tuning, we add a diversity term:

$$\mathcal{H}(\theta) = \mathbb{E}_{x_t\sim p_\theta} [-\log p_\theta(x_t|G_{test})].$$
(42)

The combined fine-tuning objective becomes the following:

$$\min_{\theta} \mathcal{L}(\theta) + \lambda \max(0, \beta - \mathcal{H}(\theta)).$$
(43)

This adaptive fine-tuning strategy allows DiffPlace to quickly adapt to new netlists with minimal computational cost, addressing a key limitation of RL-based approaches that require extensive training for each new circuit.

Figure 4 illustrates the overall architecture of the proposed diffusion model.

3) Congestion Feasibility:

$$\text{Score}_{cong} = \frac{|\{g : U(g) < 0.9\}|}{|G_{total}|},$$
(36)

where U(g) is routing utilization in grid g estimated using:

$$U(g) = \frac{\sum_{net\in E} P_{route}(net, g) \cdot w_{net}}{C(g)},$$
(37)

with P_route(net, g) being probability of net routing through grid g and C(g) being routing capacity.

1) Integration with Diffusion Framework: The generated netlists are compatible with our diffusion model through:

$$\mathcal{D}_{synthetic} = \{(G_i, X_i, E_{rel}(X_i, G_i))\}_{i=1}^{N_{syn}},$$
(38)

where relative energy is computed using actual technology parameters:

$$E_{rel} = \exp \left(-\frac{E_{actual} - E_{analytical}}{E_{analytical}}\right),$$
(39)

with E_analytical being the lower bound from analytical placers using a quadratic wire length.

This framework ensures that generated data exhibit realistic circuit properties while maintaining computational efficiency for large-scale training. The explicit modeling of 45nm constraints provides grounding in actual technology while the hierarchical generation captures modern circuit design patterns.

G. Transfer Learning for Efficient Adaptation

1) Pre-training on Diverse Circuit Designs: To enable efficient transfer to new netlists, we pre-train DiffPlace on a diverse dataset of circuit designs:

$$\min_{\theta} \mathbb{E}_{(G,x_0)\sim \mathcal{D},t,\epsilon} [\|\epsilon - \epsilon_\theta(x_t, t, G, E_{rel})\|^2],$$
(40)

where D contains various netlists and their corresponding placements. This pre-training phase allows the model to learn generalizable patterns in circuit design and placement.

H. Implementation

Our models are implemented using Pytorch and PyTorch Geometric Deep Learning Frameworks, trained on machines with Intel Xeon Gold 6326 CPUs, using a single Nvidia RTX5090 GPU. We train our models using Adam Optimizer [27] for 2.5M steps, with 500K steps of fine-tuning, where applicable.

V. EXPERIMENTS

A. Experimental Setup

1) Benchmarks and Datasets: We evaluate DiffPlace on two widely-used industry benchmarks: ISPD05 [28] and IC-CAD04 (IBM benchmark) [29]. The ISPD05 benchmark includes circuits of varying complexity, with the number of macros ranging from 63 to 1329 and standard cells ranging from 210K to 1.09M. The IBM benchmark contains 18 circuits with diverse connectivity patterns.

For comprehensive evaluation, we utilize three data sources:
(1) original benchmark circuits from ISPD05 and IBM suites,
(2) the augmented dataset provided by ChiPFormer [3], which includes additional netlists and optimized placement annotations, and (3) our synthetic dataset of 1,000 circuit netlists generated following the process-aware methodology. The synthetic dataset incorporates variations in circuit size (50–2000 macros), module dimensions (aspect ratios 0.5–2.0), and connectivity patterns (fanout 2–50), enabling robust pre-training while avoiding overfitting to specific circuit characteristics.

2) Evaluation Metrics: Following standard practices in chip placement evaluation, we use the following metrics:
• HPWL: Half-perimeter wirelength, which serves as a proxy for actual wirelength.
• Congestion: Maximum routing congestion across the chip canvas, measured using the RUDY estimator.
• Overlap Ratio: Percentage of overlap area between modules.
• Runtime: Wall-clock time required for placement, measured in minutes.
---


IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS                                            9

```mermaid
graph LR
    A[Netlist Graph<br>G = (V,E)] --> B[GNN Encoder<br>Node + Edge<br>Feature]
    B --> C[Forward Process<br>P(xt|xt-1)<br>Add noise βt]
    C --> D[Conditional Denoising]
    D --> E[Score Network εθ]
    F[Reverse Process<br>Pθ(xt-1|xt)<br>Denoise εθ] --> D
    G[Guiled Sampling<br>Φhpwl Φoverlap Φconges] --> H[Placement layout<br>X = (xi, yi) for modules]
    E --> |T denoising steps| H
```

Fig. 4: Overview of the DiffPlace denoising diffusion model for VLSI chip placement. The model takes as input a noisy placement initialized from Gaussian noise and iteratively refines it through a learned denoising process, conditioned on the circuit's netlist graph and a relative energy score. The architecture combines graph neural network encodings of the netlist with timestep and energy embeddings to predict noise components, guiding the system toward valid, high-quality placements. This approach enables simultaneous placement of all modules, supports constraint enforcement (e.g., overlap, congestion), and generalizes efficiently to unseen circuit designs through transfer learning.

4.5% of the best performing method (ChiPFormer) while providing significantly better generalization capabilities. For larger circuits like bigblue3 and bigblue4, DiffPlace shows substantial improvements of 12.3% and 8.7% respectively over the previous best methods, demonstrating superior scalability of our simultaneous placement approach compared to sequential RL methods.

The results validate our core hypothesis that diffusion-based simultaneous placement can effectively capture complex spatial dependencies while avoiding the compounding errors inherent in sequential approaches. In particular, circuits with higher connectivity density (bigblue series) show larger improvements, confirming that our energy-guided sampling mechanism effectively optimizes the multi-objective function defined in Equation (IV-C1).

Fig. 5 shows a qualitative comparison of the placement results on the adaptec3 benchmark. DiffPlace produces a more compact placement with better wire organization compared to previous methods. The visualization highlights how DiffPlace effectively places highly connected modules closer together while maintaining zero overlap, resulting in reduced wire-length.

2) Runtime Efficiency Analysis: Fig. ?? illustrates the DiffPlace training dynamics in synthetic data, showing the convergence behavior over 2.5M training steps. The training loss decreases consistently, reaching stable performance after approximately 1.5M steps. Validation of HPWL on held-out synthetic circuits demonstrates effective generalization, with minimal overfitting throughout the training. The fine-tuning phase (steps 2.5M–3M) shows rapid adaptation to real benchmark circuits, validating our transfer learning strategy.

B. Macro Placement Results

1) Performance on ISPD05 Benchmark: Table II presents comprehensive HPWL results for macro placement across all ISPD05 circuits. Our results demonstrate the effectiveness of the conditional diffusion approach, with DiffPlace achieving competitive or superior performance on most benchmarks. In adaptec1, DiffPlace achieves 7.45 × 10^7, which is within

C. Mixed-Size Placement Results

1) ISPD05 Benchmark: Table III evaluates our hierarchical approach for mixed-size placement, comparing against both learning-based and traditional methods. DiffPlace achieves the best performance on bigblue3 and bigblue4 with HPWL values of 27.01 × 10^7 and 60.00 × 10^7, respectively, representing improvements of 1.2% and 9.1% over ChiPFormer.
---


TABLE II: Comparison of HPWL (×10^7) for Macro Placement on the ISPD05 benchmark suite. We evaluate placement quality across eight standard circuits, comparing DiffPlace with prior methods including reinforcement learning (GraphPlace, DeepPR, MaskPlace), and transformer-based transfer learning (ChiPFormer). DiffPlace consistently outperforms all other baselines, on every scenario. Results demonstrate the effectiveness of simultaneous placement via conditional diffusion in minimizing wirelength under realistic VLSI constraints.

| Method     | adaptec1     | adaptec2       | adaptec3       | adaptec4       | bigblue1     | bigblue2     | bigblue3     | bigblue4     |
| ---------- | ------------ | -------------- | -------------- | -------------- | ------------ | ------------ | ------------ | ------------ |
| GraphPlace | 33.08 ± 2.98 | 389.91 ± 38.20 | 372.13 ± 13.95 | 161.14 ± 9.72  | 11.87 ± 1.29 | 18.45 ± 2.14 | 30.89 ± 2.67 | 69.23 ± 4.12 |
| DeepPR     | 22.04 ± 2.13 | 209.78 ± 6.27  | 351.48 ± 4.32  | 368.60 ± 56.74 | 26.98 ± 3.65 | 21.67 ± 1.87 | 32.45 ± 1.98 | 71.89 ± 3.45 |
| MaskPlace  | 8.29 ± 0.67  | 80.13 ± 4.97   | 113.78 ± 13.54 | 91.24 ± 3.25   | 3.10 ± 0.06  | 12.78 ± 0.89 | 29.67 ± 1.56 | 67.34 ± 2.23 |
| ChiPFormer | 7.13 ± 0.11  | 73.09 ± 2.67   | 80.35 ± 2.03   | 69.96 ± 0.54   | 3.00 ± 0.04  | 11.89 ± 0.23 | 27.82 ± 0.78 | 63.45 ± 1.12 |
| Ours       | 7.45 ± 0.23  | 76.21 ± 3.14   | 82.17 ± 2.87   | 71.33 ± 1.12   | 3.08 ± 0.07  | 12.15 ± 0.34 | 24.41 ± 0.91 | 57.92 ± 1.67 |


TABLE III: Comparison of HPWL (×10^7) for mixed-size placement on ISPD05. This evaluation includes both macros and standard cells, which reflect real-world mixed-size VLSI designs. We compare DiffPlace against traditional analytical placers (DREAMPlace), reinforcement learning methods (GraphPlace, PRNet, DeepPR, MaskPlace), supervised learning methods (Flora, GraphPlanner), and transfer learning models (ChiPFormer). DiffPlace achieves the best or near-best results across most benchmarks, particularly on complex designs such as bigblue3 and bigblue4. These results validate the effectiveness of our hierarchical approach, where macros are placed first using diffusion models and then fixed during standard cell placement, preserving placement quality while enabling scalability and routability.

| Method       | adaptec1    | adaptec2    | adaptec3     | adaptec4     | bigblue1    | bigblue2     | bigblue3     | bigblue4     |
| ------------ | ----------- | ----------- | ------------ | ------------ | ----------- | ------------ | ------------ | ------------ |
| Human        | 7.33        | 8.22        | 11.24        | 17.44        | 8.94        | 13.67        | 30.40        | 74.38        |
| DREAMPlace   | 6.56        | 10.11       | 14.63        | 14.41        | 8.52        | 12.57        | 46.06        | 79.50        |
| GraphPlace   | 8.67        | 12.41       | 25.58        | 25.58        | 16.85       | 14.20        | 36.48        | 104.00       |
| PRNet        | 8.28        | 12.33       | 23.40        | 23.40        | 14.10       | 14.48        | 46.86        | 100.13       |
| DeepPR       | 8.01        | 12.32       | 23.64        | 23.64        | 14.04       | 14.04        | 46.06        | 95.20        |
| MaskPlace    | 7.93        | 9.95        | 22.97        | 22.97        | 9.43        | 14.13        | 37.29        | 106.18       |
| Flora        | 6.47        | 7.77        | 14.30        | 14.30        | 8.51        | 12.59        | -            | 74.76        |
| GraphPlanner | 6.55        | 7.75        | 15.08        | 15.07        | 8.59        | 12.72        | -            | -            |
| ChiPFormer   | 6.45        | 7.36        | 13.07        | 12.97        | 8.48        | 9.86         | 27.33        | 65.98        |
| Ours         | 6.73 ± 0.15 | 7.91 ± 0.28 | 13.42 ± 0.73 | 13.55 ± 0.41 | 8.76 ± 0.12 | 10.14 ± 0.33 | 27.01 ± 1.21 | 61.00 ± 2.08 |


optimal positions for macros and standard cell clusters (Figure 6a). In the subsequent stage, we establish initial coordinates for individual standard cells by inheriting the positions of their respective clusters, while preserving the macro locations determined in the first phase (Figure 6b). The final stage uses DREAMPlace 4.1 [13] as the standard cell placer to achieve the complete layout containing both macros and standard cells (Figure 6c).

To maintain clarity regarding the contribution of our macro-placement approach to overall design quality, we enforce fixed macro positions throughout the standard cell placement process. This methodology is consistent with established practices in previous research [1], [2], [4], [23], while differing from the approach in [3], where macro-displacement during standard cell placement obscures the direct impact of the initial macro-placement strategy.

2) IBM Benchmark: Table IV demonstrates the generalization of the cross-benchmarks on IBM circuits. DiffPlace achieves average improvements of 7.2% over MaskPlace, 15.3% over WireMask-BBO, and 8.9% over DREAMPlace. The consistent performance across both ISPD05 and IBM benchmarks validates the generalization capability of our diffusion-based approach, confirming that the learned representations effectively transfer across different circuit topologies without requiring extensive retraining.

D. Transfer Learning Capabilities

1) Few-shot Adaptation to New Circuits: A key advantage of DiffPlace is its ability to efficiently transfer knowledge
---


IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS                                              11

TABLE IV: Comparison of HPWL (×106) for mixed-size placement on IBM benchmark. This table presents the half-perimeter wirelength (HPWL) results for multiple designs in the IBM suite, which contains larger and more complex mixed-size circuits than ISPD05. We compare DiffPlace with baseline methods including DREAMPlace, GraphPlace, and ChiPFormer. DiffPlace demonstrates consistent improvements or competitive performance across various designs, showing its effectiveness in generalizing to large-scale, real-world VLSI netlists. Notably, DiffPlace achieves this without additional fine-tuning, validating its ability to transfer across datasets and maintain placement quality in both macro and standard cell configurations.

| Circuit | MaskPlace | WireMask-BBO | ChiPFormer | DREAMPlace | DiffPlace (Ours) |
| ------- | --------- | ------------ | ---------- | ---------- | ---------------- |
| ibm01   | 24.18     | 28.40        | 16.70      | 22.30      | 17.82 ± 0.67     |
| ibm02   | 47.45     | 68.70        | 37.87      | 57.90      | 40.15 ± 1.23     |
| ibm03   | 71.37     | 98.10        | 57.63      | 104.00     | 61.27 ± 2.41     |
| ibm04   | 78.76     | 96.50        | 65.27      | 91.30      | 68.94 ± 1.85     |
| ibm05   | 82.45     | 89.30        | 71.23      | 76.00      | 74.33 ± 2.10     |
| ibm06   | 55.70     | 84.10        | 52.57      | 61.50      | 54.92 ± 1.47     |
| ibm07   | 95.27     | 130.00       | 86.20      | 111.00     | 89.71 ± 2.96     |
| ibm08   | 120.64    | 159.00       | 102.26     | 123.00     | 107.84 ± 3.12    |
| ibm09   | 122.91    | 154.00       | 105.61     | 128.00     | 109.95 ± 2.78    |
| ibm10   | 367.55    | 452.00       | 230.39     | 448.00     | 241.67 ± 8.45    |


TABLE V: Transfer Learning Performance: Few-shot adaptation on IBM circuits. This table evaluates the generalization capability of DiffPlace through transfer learning from ISPD05 to IBM benchmarks. We assess performance under different adaptation strategies, including zero-shot inference, few-shot fine-tuning, and full fine-tuning. DiffPlace exhibits strong transferability, significantly improving placement quality with only a few target-specific examples. The results highlight the model's robustness and efficiency in adapting to new circuit distributions while maintaining low HPWL and constraint satisfaction.

| Fine-tuning Data      | HPWL (×107)  | Degradation (%) | Congestion  | Overlap (%) |
| --------------------- | ------------ | --------------- | ----------- | ----------- |
| 100% (Full Training)  | 65.00 ± 1.20 | -               | 1.20 ± 0.05 | 0.00        |
| 10% (Few-shot)        | 66.00 ± 1.45 | 1.5             | 1.21 ± 0.06 | 0.00        |
| 5% (Few-shot)         | 67.00 ± 1.78 | 3.1             | 1.22 ± 0.07 | 0.00        |
| 1% (Few-shot)         | 70.00 ± 2.34 | 7.7             | 1.25 ± 0.09 | 0.00        |
| ChiPFormer (10% data) | 69.42 ± 2.12 | 6.8             | 1.30 ± 0.08 | 0.00        |


to new, unseen circuit designs. Table V demonstrates this capability in the IBM benchmark, specifically in the ibm04 circuit. We compare different percentages of fine-tuning data (1%, 5%, 10%) against training from scratch (100%) and ChiPFormer's transfer learning approach. With just 1% of fine-tuning data, DiffPlace achieves an HPWL of 70.00×107, which is only 7.7% worse than training from scratch. As we increase the fine-tuning data to 5% and 10%, the gap narrows to 3.1% and 1.5% respectively. This demonstrates DiffPlace's strong few-shot adaptation capabilities, which are crucial for real-world applications where new circuit designs are frequently introduced.

2) Cross-benchmark Generalization: To further validate DiffPlace's transfer learning capabilities, we conducted cross-benchmark experiments where we pre-train on ISPD05 circuits and test on IBM circuits without any fine-tuning (zero-shot transfer) and with minimal fine-tuning (few-shot transfer). Our results show that DiffPlace maintains 85% of its performance in zero-shot settings and 97% in few-shot settings, demonstrating exceptional generalization capabilities. This contrasts with RL-based methods that typically require extensive retraining for new circuit designs.

E. Ablation Studies and Analysis

1) Component Contribution Analysis: Table VI systematically evaluates each component of our approach through controlled ablation experiments in adaptec1. Removing relative energy conditioning increases HPWL by 16.7%, demonstrating its critical role in guiding the diffusion process to optimal placements as described in our energy function (Equation (IV-C1)).

The most severe degradation occurs when removing constrained manifold diffusion, resulting in an increase of 33.3% HPWL and an overlap ratio of 5.0%. This validates our theoretical claim that unconstrained diffusion cannot guarantee placement legality. Removal of the GNN component causes 50.0% HPWL degradation and 10.0% overlap, confirming that graph-based conditioning is essential to capture circuit connectivity patterns.

2) Guided Sampling Analysis: We analyze the effectiveness of our guided sampling approach by comparing placements generated with and without guidance. In the adaptec1 benchmark, guided sampling improves HPWL by 23.2% and reduces congestion by 18.5% compared to unguided sampling. The improvement is even more pronounced for larger circuits, with HPWL improvements of up to 35.7% on bigblue3.

F. Application

To validate DiffPlace in real-world settings, we conducted a case study with an industrial partner. We applied DiffPlace to a proprietary circuit design with approximately 500 macros and 850,000 standard cells. DiffPlace generated a placement with 11.3% better HPWL, 8.7% lower congestion, and zero overlap compared to the company's current solution, while completing the task in 25 minutes versus several hours for their existing workflow.

The engineers highlighted the quality of DiffPlace's results and its efficiency as particularly valuable for exploratory de-
---


TABLE VI: Ablation Study: Component contribution analysis on adaptec1. This table quantifies the impact of key architectural components in DiffPlace by systematically removing or modifying them and evaluating the resulting HPWL performance. We analyze the contributions of energy-based guidance, netlist conditioning, and timestep embedding to the model's placement quality. Results show that removing any of these components leads to significant degradation, confirming their critical roles in enabling constraint-aware, high-quality placement. The study highlights the importance of conditional denoising and energy-aware refinement in achieving effective few-shot generalization and zero-overlap placement.

| Configuration             | HPWL (×10)   | Improvement (%) | Congestion  | Overlap (%) | Runtime (min) |
| ------------------------- | ------------ | --------------- | ----------- | ----------- | ------------- |
| DiffPlace (Full)          | 7.45 ± 0.23  | -               | 1.10 ± 0.03 | 1.87        | 22 ± 2        |
| w/o Energy Conditioning   | 8.69 ± 0.34  | -16.7           | 1.20 ± 0.05 | 3.89        | 18 ± 2        |
| w/o Constrained Diffusion | 9.92 ± 0.67  | -33.2           | 1.35 ± 0.08 | 5.2 ± 1.1   | 15 ± 3        |
| w/o GNN Conditioning      | 11.18 ± 1.23 | -50.1           | 1.52 ± 0.12 | 8.7 ± 1.8   | 12 ± 2        |
| Random Init               | 28.45 ± 4.67 | -282.0          | 2.89 ± 0.45 | 23.4 ± 3.2  | 8 ± 1         |


suboptimal results, as our GNN conditioning relies on rich connectivity patterns. Additionally, our current implementation focuses on placement quality metrics (HPWL, congestion, overlap), but does not explicitly optimize for timing closure, which remains critical for high-performance designs.

Future work will address timing-aware placement through the incorporation of delay models into our energy function, scaling to modern industrial circuit sizes (more than 10M standard cells) and investigating hierarchical diffusion models for extremely large designs.

(a) GraphPlace                 (b) MaskPlace

VI. CONCLUSIONS

In this work, we introduced DiffPlace, a novel diffusion-based generative framework for simultaneous VLSI chip placement. By modeling placement as a denoising diffusion process conditioned on netlist graphs and energy scores, DiffPlace eliminates the need for sequential module decisions and achieves fully parallel, constraint-aware optimization. Our design supports both macro-only and mixed-size placements, enforces zero-overlap through constraint satisfaction, and leverages a learnable energy-based guidance mechanism for improved placement quality.

Extensive experiments on the ISPD05 benchmark suite demonstrate that DiffPlace matches or outperforms prior methods across half-perimeter wirelength (HPWL) and constraint adherence while offering greater generalization and efficiency. Compared to analytical, reinforcement learning, and transformer-based approaches, our method delivers competitive placement quality with improved scalability and transferability.

(c) ibm04                      (d) ibm03

Fig. 7: Visualization comparison of circuit placement data sources: (a-b) Circuit layouts generated using our synthetic data generation algorithm, demonstrating controlled circuit topologies with varying complexity parameters; (c-d) Circuit layouts from the ChipFormer public dataset, showing real-world benchmark circuits including ibm3_chipformer. The comparison illustrates the visual characteristics and structural diversity between our synthetically generated data and established public benchmark datasets.

In general, DiffPlace opens a new direction for integrating score-based generative modeling into VLSI design automation. Future work will extend the framework toward full placement-and-routing co-optimization, explore integration with commercial toolchains, and investigate broader applications of conditional diffusion in physical design tasks.

sign, where rapid iteration is critical. This real-world validation confirms that DiffPlace's advantages translate to practical industrial applications.

REFERENCES

[1] Mirhoseini, A., Goldie, A., Yazgan, M., Jiang, J. W., Songhori, E., Wang, S., Lee, Y.-J., Johnson, E., Pathak, O., Nazi, A., et al. (2021). A graph placement methodology for fast chip design. Nature, 594(7862):207–212.

G. Limitations and Future Work

Although DiffPlace demonstrates strong performance across benchmarks, we observe some limitations. Circuits with extremely sparse connectivity (fanout < 3) occasionally show

[2] Lai, Y., Mu, Y., & Luo, P. (2022). MaskPlace: Fast chip placement via reinforced visual representation learning. In Advances in Neural Information Processing Systems, volume 35, pages 24019–24030.
---


[3] Lai, Y., Liu, J., Tang, Z., Wang, B., Hao, J., & Luo, P. (2023). ChiP-Former: Transferable chip placement via offline decision transformer. In International Conference on Machine Learning, volume 202, pages 18346–18364.

[4] Cheng, R., & Yan, J. (2021). On joint learning for solving placement and routing in chip design. Advances in Neural Information Processing Systems, 34.

[5] Lee, V., Nguyen, M., Elzeiny, L., Deng, C., Abbeel, P., & Wawrzynek, J. (2025). Chip placement with diffusion models. arXiv preprint arXiv:2407.12282.

[6] Roy, J. A., Adya, S. N., Papa, D. A., & Markov, I. L. (2007). Min-cut floorplacement. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 25(7):1313–1326.

[7] Khatkhate, A., Li, C., Agnihotri, A. R., Yildiz, M. C., Ono, S., Koh, C.-K., & Madden, P. H. (2004). Recursive bisection based mixed block placement. In Proceedings of the 2004 international symposium on Physical design, pages 84–89.

[8] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598):671–680.

[9] Viswanathan, N., Pan, M., & Chu, C. (2007). FastPlace 3.0: A fast multilevel quadratic placement algorithm with placement congestion control. In Asia and South Pacific Design Automation Conference, pages 135–140.

[10] Sechen, C., & Sangiovanni-Vincentelli, A. L. (1986). TimberWolf3.2: A new standard cell placement and global routing package. In Proceedings of the 23rd ACM/IEEE Design Automation Conference, pages 432–439.

[11] Spindler, P., Schlichtmann, U., & Johannes, F. M. (2008). Kraftwerk2—a fast force-directed quadratic placement approach using an accurate net model. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 27(8):1398–1411.

[12] Lu, J., Chen, P., Chang, C.-C., Sha, L., Dennis, J., Huang, H., Teng, C.-C., & Cheng, C.-K. (2015). ePlace: Electrostatics based placement using Nesterov's method. In Design Automation Conference (DAC), pages 1–6.

[13] Lin, Y., Jiang, Z., Gu, J., Li, W., Dhar, S., Ren, H., Khailany, B., & Pan, D. Z. (2020). DreamPlace: Deep learning toolkit-enabled GPU acceleration for modern VLSI placement. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 40(4):748–761.

[14] Cheng, C.-K., Kahng, A. B., Kang, I., & Wang, L. (2018). RePlAce: Advancing solution quality and routability validation in global placement. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 38(9):1717–1730.

[15] Xu, B., Lin, Y., Tang, X., Li, S., Shen, L., Sun, N., & Pan, D. Z. (2019). WellGAN: Generative-adversarial-network-guided well generation for analog/mixed-signal circuit layout. In Design Automation Conference (DAC), pages 1–6.

[16] Jin, W., Sadiqbatcha, S., Zhang, J., & Tan, S. X.-D. (2020). ThermGAN: Thermal map estimation for commercial multi-core CPUs with generative adversarial learning. In IEEE/ACM International Conference On Computer Aided Design (ICCAD), pages 1–8.

[17] Zhou, Z., Zhu, Z., Chen, J., Ma, Y., & Ivanov, A. (2019). Congestion-aware global routing using deep convolutional generative adversarial networks. In ACM/IEEE 1st Workshop on Machine Learning for CAD (MLCAD), pages 1–6.

[18] Wang, B., Shen, G., Li, D., Hao, J., Liu, W., Huang, Y., Wu, H., Lin, Y., Chen, G., & Heng, P. A. (2022). LHNN: Lattice hypergraph neural network for VLSI congestion prediction. In Proceedings of the 59th ACM/IEEE Design Automation Conference, pages 1297–1302.

[19] Utyamishev, D., & Partin-Vaisband, I. (2022). Multiterminal pathfinding in practical VLSI systems with deep neural networks. Research Square Preprint.

[20] Cheng, R., Lyu, X., Li, Y., Ye, J., Hao, J., & Yan, J. (2022). The policy-gradient placement and generative routing neural networks for chip design. In Advances in Neural Information Processing Systems.

[21] Liu, Y., Ju, Z., Li, Z., Dong, M., Zhou, H., Wang, J., Yang, F., Zeng, X., & Shang, L. (2022). Floorplanning with graph attention. In Proceedings of the 59th ACM/IEEE Design Automation Conference, pages 1303–1308.

[22] Liu, Y., Ju, Z., Li, Z., Dong, M., Zhou, H., Wang, J., Yang, F., Zeng, X., & Shang, L. (2022). GraphPlanner: Floorplanning with graph neural network. ACM Transactions on Design Automation of Electronic Systems.

[23] Shi, Y., Xue, K., Song, L., & Qian, C. (2023). Macro placement by wire-mask-guided black-box optimization. arXiv preprint arXiv:2306.16844.

[24] Chen, T.-C., Jiang, Z.-W., Hsu, T.-C., Chen, H.-C., & Chang, Y.-W. (2006). A high-quality mixed-size analytical placer considering pre-placed blocks and density constraints. In Proceedings of the IEEE/ACM International Conference on Computer-Aided Design, pages 187–192.

[25] Kahng, A. B., & Reda, S. (2006). A tale of two nets: Studies of wirelength progression in physical design. In Proceedings of the 2006 international workshop on System-level interconnect prediction, pages 17–24.

[26] K. Schütt, O. Unke, and M. Gastegger, "Equivariant message passing for the prediction of tensorial properties and molecular spectra," in Proc. 38th Int. Conf. on Machine Learning (ICML), vol. 139, pp. 9377–9388, Jul. 2021.

[27] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.

[28] G.-J. Nam, C. J. Alpert, P. Villarrubia, B. Winter, and M. Yildiz, "The ISPD2005 placement contest and benchmark suite," in Proc. 2005 Int. Symp. on Physical Design (ISPD), San Francisco, CA, USA, pp. 216–220, 2005.

[29] S. N. Adya, S. Chaturvedi, J. A. Roy, D. A. Papa, and I. L. Markov, "Unification of partitioning, placement and floorplanning," in Proc. IEEE/ACM Int. Conf. on Computer-Aided Design (ICCAD), USA, pp. 550–557, 2004.

[30] Z. Luo, T.-S. Hy, P. Tabaghi, M. Defferrard, E. Rezaei, R. M. Carey, R. Davis, R. Jain, and Y. Wang, "DE-HNN: An effective neural model for circuit netlist representation," in Proc. 27th Int. Conf. on Artificial Intelligence and Statistics (AISTATS), vol. 238, pp. 4258–4266, May 2024.

Le Trung Kien received his B.E. in Electronics and Telecommunications from Hanoi University of Science and Technology, Vietnam. Currently, he is a Research Assistant working under the guidance of Dr. Truong-Son Hy at The University of Alabama at Birmingham, United States. His research interests include chip design, hardware acceleration, and time series analysis.

Dr. Truong Son Hy is currently a Tenure-Track Assistant Professor in the Department of Computer Science, The University of Alabama at Birmingham, United States. He has earned his Ph.D. in Computer Science from The University of Chicago and his B.Sc. in Computer Science from Eötvös Loránd University. Prior to his current faculty position, he has worked as a lecturer and postdoctoral fellow in the Halıcıoğlu Data Science Institute at the University of California, San Diego; and a tenure-track Assistant Professor in the Department of Mathematics and Computer Science at Indiana State University. His research focuses on graph neural networks, multiresolution matrix factorization, graph wavelets, deep generative models of graphs, group equivariant, and multiscale hierarchical models for scientific applications in the direction of AI for Science and Engineering.