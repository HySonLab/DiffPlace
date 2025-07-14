

# Chip Placement with Diffusion Models

Vint Lee<sup>1</sup> Minh Nguyen<sup>1</sup> Leena Elzeiny<sup>2</sup> Chun Deng<sup>3</sup> Pieter Abbeel<sup>1</sup> John Wawrzynek<sup>1</sup>

## Abstract

Macro placement is a vital step in digital circuit design that defines the physical location of large collections of components, known as macros, on a 2D chip. Because key performance metrics of the chip are determined by the placement, optimizing it is crucial. Existing learning-based methods typically fall short because of their reliance on reinforcement learning (RL), which is slow and struggles to generalize, requiring online training on each new circuit. Instead, we train a diffusion model capable of placing new circuits zero-shot, using guided sampling in lieu of RL to optimize placement quality. To enable such models to train at scale, we designed a capable yet efficient architecture for the denoising model, and propose a novel algorithm to generate large synthetic datasets for pre-training. To allow zero-shot transfer to real circuits, we empirically study the design decisions of our dataset generation algorithm, and identify several key factors enabling generalization. When trained on our synthetic data, our models generate high-quality placements on unseen, realistic circuits, achieving competitive performance on placement benchmarks compared to state-of-the-art methods.

## 1. Introduction

Placement is an important step of digital hardware design where collections of small components, such as logic gates (standard cells), and large design blocks, such as memories, (macros) are arranged on a 2-dimensional physical chip based on a connectivity graph (netlist) of the components. Because the physical layout of objects determines the length of wires (and where they can be routed), this step has a significant impact on key metrics, such as power consumption, performance, and area of the produced chip. In particular, the placement of macros, which is the focus of our work, is especially important because of their large size and high connectivity relative to standard cells.

Traditionally, macro placement is done with commercial tools such as Innovus from Cadence, which requires input from human experts. The process is also time-consuming and expensive. On the other hand, the use of ML techniques shows promise in automating this process, as well as creating better-optimized placements than commercial tools, which rely heavily on heuristics.

Even so, existing works mostly rely on reinforcement learning (RL) (Mirhoseini et al., 2020; Cheng & Yan, 2021; Lai et al., 2022; 2023; Gu et al., 2024), an approach with several key limitations. First, RL is challenging to scale — it is sample-inefficient, and has difficulty generalizing to new problems. Many of these methods, for instance, treat each new circuit as a separate task, training a new agent from scratch for every new netlist (Cheng & Yan, 2021; Lai et al., 2022; Gu et al., 2024). Despite efforts to mitigate this by incorporating offline pre-training, the scarcity of publicly-available data means that such methods struggle to generalize, and still require a significant amount of additional training for each new netlist (Mirhoseini et al., 2020; Lai et al., 2023). Second, by casting placement as a Markov Decision Process (MDP), these works require agents to learn a sequential placement of objects (standard cells or macros), which creates challenges when suboptimal choices near the start of the trajectory cannot be reversed.

To circumvent these issues, we instead adopt a different approach: leveraging powerful generative models, in particular diffusion models, to produce near-optimal chip placements for a given netlist. Diffusion models address the weaknesses of RL approaches because they can be trained offline at scale, then used zero-shot on new netlists, simultaneously placing all objects as shown in Figure 1. Moreover, our approach takes advantage of the great strides made in techniques for training and sampling diffusion models, such as guided sampling (Dhariwal & Nichol, 2021; Bansal et al., 2023), to achieve better results.

Training a large and generalizable diffusion model, however, comes with its own challenges. First, the vast majority of circuit designs and netlists of interest are proprietary, severely limiting the quality and quantity of available train-
---


# Chip Placement with Diffusion Models

ing data. Second, many of these circuits are large, containing hundreds of thousands of macros and cells. The denoising model used must therefore be computationally efficient and scalable, in addition to working well within the noise-prediction framework.

Our work addresses these challenges, and we summarize our main contributions as follows:

**Synthetic Data Generation** We present a method for easily generating large amounts of synthetic netlist and placement data. Our insight is that the inverse problem — producing a plausible netlist such that a given placement is near-optimal — is much simpler to solve. This allows us to produce data without the need for commercial tools or higher-level design specifications.

**Dataset Design** We conduct an extensive empirical study investigating the generalization properties of models trained on synthetic data, identifying several factors, such as the scale parameter, that cause models to generalize poorly. We use these insights to design synthetic datasets that allow for effective zero-shot transfer to real circuits.

**Model Architecture** We propose a novel neural network architecture with interleaved graph convolutions and attention layers to obtain a model that is both computationally efficient and expressive.

By combining these ingredients, our method can generate placements for unseen netlists in a zero-shot manner, achieving results competitive with state-of-the-art on the ICCAD04 benchmark (Adya et al., 2004), also known as the IBM benchmark. Remarkably, our model accomplishes this without ever having trained on real-world circuit data.

## 2. Related Work

Google's RL-based approach, CircuitTraining (Mirhoseini et al., 2021; Yue et al., 2022), employs a graph neural network (GNN) to generate netlist embeddings for multiple RL agents. While this method demonstrated state-of-the-art results, it relies on an initial placement generated using other methods (Cheng et al., 2023), and requires computationally expensive online training on new circuits. Several RL approaches follow to improve on runtime (Cheng & Yan, 2021; Lai et al., 2022; Gu et al., 2024), macro ordering (Chen et al., 2023), and proxy cost predictions (Zheng et al., 2023a;b; Wang et al., 2022; Ghose et al., 2021). Recently, ChiPFormer (Lai et al., 2023) demonstrated strong results on several benchmarks, combining offline and online RL to improve generalization. However their method still trains on in-distribution examples, and requires hours of online training on each new netlist for optimal results.

In contrast, Flora (Liu et al., 2022a) and GraphPlanner (Liu et al., 2022b) deviate from sequential placement formulations by leveraging a variational autoencoder (VAE) (Kingma & Welling, 2022) to generate placements. Flora further introduces a synthetic data generation scheme; however, it lacks variation in object sizes and restricts connections to only the nearest neighbors, which, as our experiments indicate, limits generalization to realistic circuit layouts (see Section 5.1). Furthermore, these generative models struggle to learn the underlying distribution of legal placements, frequently producing overlapping results.

Other approaches avoid the use of machine learning altogether. WireMask-BBO (Shi et al., 2023) utilizes black-box optimization algorithms to find optimal macro placements over continuous coordinates, while legalizing and evaluating the solution quality on a discrete grid. However, their usage of black-box optimization, such as evolutionary algorithms, leads to lengthy search times that must be started from scratch for each new circuit.

## 3. Background

### 3.1. Problem Statement

Our goal is to train a diffusion model to sample from $f(x|c)$, where the placement $x$ is a set of 2D coordinates for each object and the netlist $c$ describes how the objects are connected in a graph, as well as the size of each object. We normalize the coordinates to the chip boundaries, so that they are within $[-1, 1]$.

We represent the netlist as a graph $(V, E)$ with node and edge attributes ${p_i}_{i∈V}$ and ${q_{ij}}_{(i,j)∈E}$. We define $p_i$ to be a 2D vector describing the normalized height and width of the object, while $q_{ij}$ is a 4D vector containing the positions of the source and destination pins, relative to the center of their parent object. We convert the netlist hypergraph into this representation by connecting the driving pin of each netlist to the others with undirected edges. This compact representation contains all the geometric information needed for placement, and allows us to leverage the rich body of existing GNN methods.

### 3.2. Evaluation Metrics

To evaluate generated placements, we use legality, which measures how easily the placement can be used for downstream tasks (eg. routing); and half-perimeter wire length (HPWL), which serves as a proxy for chip performance.

While a legal placement has to satisfy other criteria, in this work we focus on a simpler, commonly used constraint (Mirhoseini et al., 2020; Lai et al., 2023): the objects cannot overlap one another, and must be within the bounds of the canvas. We can therefore define legality score as $A_u/A_s$,
---


# Chip Placement with Diffusion Models

Figure 1. Denoising process for generating placements. In contrast to RL approaches, our method places all objects simultaneously. The middle 4 panels show the predicted output x̂₀ at intervals of 200 steps, while the first and last panels are xT (Gaussian noise) and x₀ (generated placement).

where Au is the area, within the circuit boundary, of the union of all placed objects, and As is the sum of areas of all individual objects. A legality of 1 indicates that all constraints are satisfied.

Routed wirelength influences critical metrics because long wires create delay between components, influencing timing and power consumption. HPWL is used as an approximation to evaluate placements prior to routing (Chen et al., 2006; Kahng & Reda, 2006). Because the scale of HPWL varies greatly between circuits, for our experiments on synthetic data we report the HPWL ratio, defined for a given netlist as Wgen/Wdata, where Wgen is the HPWL for the model-generated placement, while Wdata is the HPWL of the placement in the dataset.

Our objective is therefore to generate legal placements with minimal HPWL.

## 3.3. Diffusion Models

Diffusion models (Song et al., 2021; Ho et al., 2020) are a class of generative models whose outputs are produced by iteratively denoising samples using a process known as Langevin Dynamics. In this work we use the Denoising Diffusion Probabilistic Model (DDPM) formulation (Ho et al., 2020), where starting with Gaussian noise xT, we perform T denoising steps to obtain xT−1, xT−2, . . . , x0, with the fully denoised output x0 as our generated sample. In DDPMs, each denoising step is performed according to

$$x_{t-1} = \alpha_t \cdot x_t + \beta_t \cdot \hat{\epsilon}_\theta(x_t, t, c) + \sigma_t \cdot z,   (1)$$

where αt, βt, σt are constants defined by the noise schedule, z ∼ N(0, I) is injected noise, and ε̂θ is the learned denoising model taking xt, t and context c as inputs. By training ε̂θ to predict the noise added to samples from the dataset, DDPMs are able to model arbitrarily complex data distributions.

| !Diagram showing steps of generating synthetic data | 1. Place objects
2. Generate pins
3. Compute pin distances
4. Sample edges |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |


Figure 2. Visualization of the steps involved in generating synthetic data.

## 4. Methods

### 4.1. Generating Synthetic Data

We obtain datasets (Table 1) consisting of tuples (x, c) using the method outlined below.

First, we randomly generate objects by sampling sizes uniformly and placing them at random within the circuit boundary, ensuring legality by retrying if objects overlap. Following Rent's Rule (Lanzerotti et al., 2005), we then sample a number of pins for each object using a power law.

To generate edges, we start by computing the distance l for each pair of pins on different objects, then sample independently from Bernoulli(p(l)), where p(l) ∈ [0, 1] is a distance-dependent probability which we refer to as the edge distribution¹. To approximate the structure of real circuits and bias the model towards lower HPWL placements, we choose p ∝ exp(−l/s), so that the probability of generating edges decays exponentially with L1 distance l normalized by a scale parameter s.

This simple algorithm, depicted in Figure 2, allows us to efficiently generate large numbers of training examples for our models without the using any commercial tools or design specifications. Using 32 CPUs, we are able to produce 100k "circuits" each containing approximately 200 objects in a day.

Our algorithm is also highly flexible, allowing many choices for the distributions of the number of objects, their sizes, the

¹p is not, however, a probability distribution
---


# Chip Placement with Diffusion Models

edges, scale parameters, and so on. To better understand the design space of synthetic datasets, we conduct an extensive empirical study in Section 5.1, identifying several factors that are vital for training models that transfer zero-shot to real circuits.

Based on our results, we designed 2 synthetic datasets, v1 and v2, with parameters listed in Table 1.

## 4.2. Model Architecture

We developed a novel architecture for the denoising model, shown in Figure 3. We highlight below several key elements of our design that we empirically determined (see Section 5.2) to be important for the placement task:

### Interleaved GNN and Attention Layers
We use the message-passing GNN layers for their computational efficiency in capturing node neighborhood information, while the interleaved attention (Vaswani et al., 2017) layers address the oversmoothing problem in GNNs by allowing information transfer between nodes that are distant in the netlist graph, but close on the 2D canvas. We find that combining the two types of layers is critical, and significantly outperforms using either type alone.

### MLP Blocks
We found that inserting residual 2-layer MLP blocks between each GNN and Attention block improved performance significantly for a negligible increase in computation time.

### Sinusoidal 2D Encodings
The model receives 2D sinusoidal position encodings, in addition to the original (x, y) coordinates, as input. This method improves the precision with which the model places small objects, leading to placements with better legality.

In this work, we use 3 sizes of models: Small, Medium, and Large, with 233k, 1.23M, and 6.29M parameters respectively. A full list of model hyperparameters can be found in Appendix A.

## 4.3. Guided Sampling

One key advantage of using diffusion models is the ease of optimizing for downstream objectives through guided sampling. We use backwards universal guidance (Bansal et al., 2023) with easily computed potential functions to optimize the generated HPWL and legality without training additional reward models or classifiers. The guidance potential φ(x) is defined as the weighted sum w_legality · φ_legality + w_hpwl · φ_hpwl of potentials for each of our optimization objectives.

The legality potential φ_legality(x) for a netlist with objects V is given by:

$$\phi_\text{legality}(x) = \sum_{i,j\in V} \min(0, d_{ij}(x))^2 \qquad (2)$$

where d_ij is the signed distance between objects i and j, which we can compute easily for rectangular objects. Note that the summand is 0 for any pair of non-overlapping objects, and increases as overlap increases.

We define φ_hpwl(x) simply as the HPWL of the placement x. We compute this in a parallelized, differentiable manner by casting HPWL computation in terms of the message-passing framework used in GNNs (Gilmer et al., 2017) and implementing a custom GNN layer with no learnable parameters in PyG (Fey & Lenssen, 2019).

Instead of gradients from a classifier (Dhariwal & Nichol, 2021), we use the backwards universal guidance force g(x_t) = Δ_φ x̂_0. Here, x̂_0 is the prediction of x_0 based on the denoising model's output at time step t, and Δ_φ is the φ-optimal change in x̂_0-space, computed using gradient descent. The combined diffusion score is then given by f_θ(x_t) + w_g · g(x_t). We refer the reader to Bansal et al. (2023) for more details.

In the simple implementation, w_legality and w_hpwl are set as constant hyperparameters. However, the optimal weights can vary depending on the circuit's connectivity properties. Instead, we take inspiration from constrained optimization to automatically tune the weights. To solve

$$\min_x \phi_\text{hpwl}(x) \quad \text{s.t.} \quad \phi_\text{legality}(x) = 0, \qquad (3)$$

we optimize the Lagrangian L(λ, x) = φ_hpwl(x) + λ · φ_legality(x) simultaneously with respect to x and the Lagrange multiplier λ. We instantiate this idea during guidance by performing interleaved gradient descent steps of φ_hpwl(x) + w_legality · φ_legality(x) with respect to x, and w_legality · (φ_legality(x) − ε) with respect to w_legality.

## 4.4. Training and Evaluation

Due to the lack of real placement data, we train our models entirely on synthetic data (see Section 4.1). For placing real circuit netlists, we train our models in two stages: we first train on our v1 dataset of smaller circuits, then fine-tune on v2 which contains larger circuits. Details on dataset design are provided in Section 5.1.

We evaluate the performance of our model on circuits in the publicly available ICCAD04 (Adya et al., 2004) (also known as IBM) benchmark. Because these circuits contain hundreds of thousands of small standard cells, we follow prior work (Mirhoseini et al., 2020) and cluster the standard cells into 512 partitions using hMetis (Karypis et al., 1997). Each cluster is assigned to the nets of its constituent standard
---


# Chip Placement with Diffusion Models

Block (×N)

| Encoder Sinusoidal<br/>Encodings Linear | ResGNN blockGNN	GNN |
| --------------------------------------- | ------------------- |

</td>
<td>
MLP<br>
2 layers
</td>
<td>
AttGNN block

| GNN | Attention | GNN | Attention |
| --- | --------- | --- | --------- |

</td>
<td>
MLP<br>
2 layers
</td>
</tr>
</table>

Figure 3. Diagram of our denoising model. Residual connections, edge feature inputs, nonlinearities, and normalization layers have been omitted for clarity.

cells (nets within a single cluster are removed) with pins located at the cluster center, while the size of each cluster is the total area of its standard cells.

## 4.5. Implementation

Our models are implemented using Pytorch (Paszke et al., 2019) and Pytorch-Geometric (Fey & Lenssen, 2019), and trained on machines with Intel Xeon Gold 6326 CPUs, using a single Nvidia A5000 GPU. We train our models using the Adam optimizer (Kingma & Ba, 2014) for 3M steps, with 250k steps of fine-tuning where applicable.

## 5. Experiments

### 5.1. Designing Synthetic Data

To generate synthetic datasets that allow for zero-shot transfer, we first have to understand which parameters are important for generalization, and which ones are not. We therefore investigate the generalization capabilities of our model along several axes by evaluating a single trained model on datasets generated using various parameters. By identifying parameters that the model struggles to generalize across, we can design our synthetic dataset to facilitate zero-shot transfer by ensuring that for such parameters, the synthetic distribution covers that of real circuits (see Section 5.1.4).

In this section, we evaluate a model trained on a dataset with a narrow distribution, with key parameters listed in Table 1. The full set of parameters is listed in Appendix A.

#### 5.1.1. NUMBER OF EDGES AND VERTICES

Figure 4 shows how legality changes with the number of edges in the test dataset. The model generalizes remarkably well to datasets with more edges than it was trained on, while performance degrades quickly when fewer edges are present. We hypothesize that an increased number of edges allows the GNN layers to propagate information more efficiently, improving or maintaining the model's performance.

In contrast, Figure 5 shows that the model struggles to generalize to larger circuits than it was trained on, with legality decreasing as the number of vertices increases.

#### 5.1.2. SCALE PARAMETER

In our data generation algorithm, the scale parameter s determines the expected length of generated edges. A larger s means that distant pins are more likely to be connected, while a small value of s means that only nearby pins are connected. Thus, the scale parameter has a significant impact on the properties of the graph generated, such as the number of neighbors per vertex, as well as the optimality of the corresponding placement. To understand how these effects influence the placements generated by the model, we evaluate our model, trained on data with a fixed value of s, on datasets with different scale parameters.

Figure 6 shows that the model performs well at longer scale parameters up to s = 0.4, with legality dropping sharply past it. Meanwhile, lowering s causes HPWL to worsen significantly, with the model generating placements more than 1.5× worse than the dataset.

This could be because for small s, the presence of edges between objects means that they are very likely near each other in the dataset placement, providing the model with a lot of information on where objects should be placed. If

| Figure 4. Legality decreases on circuits with fewer edges, while adding edges does not degrade performance. The training data has 1740 edges on average. | Figure 5. Legality decreases on circuits with more vertices, indicating poor generalization. The training data has 230 edges on average. |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| \`\`\` 1.00 0.98 Legality 0.96 0.94 100 1000 10000 Edges \`\`\`                                                                                          | \`\`\` 1.00 0.98 Legality 0.96 0.94 100 500 1000 Vertices \`\`\`                                                                         |

---


# Chip Placement with Diffusion Models

| Name | Circuits | Vertices | Edges | Scale Parameter (s) | p(l)              |
| ---- | -------- | -------- | ----- | ------------------- | ----------------- |
| v0   | 40000    | 230      | 1740  | 0.2                 | γ · exp (−l/s)    |
| v1   | 40000    | 230      | 1600  | ∼ log U(0.05, 1.6)  | γ(s) · exp (−l/s) |
| v2   | 5000     | 960      | 9510  | ∼ log U(0.025, 0.8) | γ(s) · exp (−l/s) |


|            | Exponential | Sigmoid  | Linear          |
| ---------- | ----------- | -------- | --------------- |
| p(l) ∝     | e(−l/s)     | σ(l − s) | max((s−x)/s, 0) |
| Legality   | 0.982       | 0.983    | 0.982           |
| HPWL Ratio | 1.003       | 1.012    | 1.003           |


Figure 6. Legality drops sharply when increasing scale parameter past a certain point, while scale parameters smaller than the training data causes HPWL to worsen significantly. The training data is generated using a scale of 0.2.

the model is then evaluated on circuits with larger s, the increased number of neighbors causes the model to place too many objects in the same vicinity, leading to clumps forming (see Figure 7) and poor legality. Conversely, when evaluated on circuits with smaller s, the model's inductive biases are not strong enough to place connected objects as close together as possible, leading to longer, worse, HPWL.

Figure 7. Increasing scale parameter causes the diffusion model, trained only on circuits with s = 0.2, to clump objects together.

## 5.1.3. DISTRIBUTION OF EDGES

To bias the model towards generating placements with low HPWL, we sample edges with probability p(l) that exponentially decays with edge length l. However, there is no guarantee that optimal placements for real circuits follow such a distribution. We therefore investigate how this choice of l affects generalization to other edge distributions p(l).

In Table 2, we see that our model performs well, both in legality and HPWL, on circuits where edges are sampled from different distributions. This is a promising indication that our training data can allow models to generalize zero-shot to unseen circuits that lie outside the training distribution.

## 5.1.4. ZERO-SHOT TRANSFER TO REAL CIRCUITS

These results allow us to design new datasets, which we refer to as v1 and v2, that better enable zero-shot transfer to real circuits. Because our models generalize poorly to larger circuits, they require training on circuits similar in size to real (clustered) circuits, which contain ∼1000 components. To satisfy this while maintaining computational efficiency, we pre-train on a large set of smaller circuits (v1) before fine-tuning on a small set of larger circuits (v2), keeping the number of edges low in both datasets. To ensure model performance across different length scales, we also train on a broad distribution of scale parameters. Finally, since an exponentially decaying p generalizes well to other distributions without sacrificing HPWL, we continue using it in our new datasets.

The parameters for generating the v1 and v2 datasets are summarized in Table 1, with a full list provided in Table 7. We find that training on these broad-distribution datasets allow for effective zero-shot transfer to the real-world circuits in the IBM benchmark, with legality increasing significantly when training on v1 and v2, compared to the narrower v0 dataset. Although our simple data generation algorithm does not capture all the nuances of real circuits, such as the multimodal distributions of both edges and object sizes, it covers the important features well enough for our model to learn to produce reasonable placements on IBM circuits, some of which are shown in Figure 8.
---


## 5.2. Model Architecture

We demonstrate the importance of several components of our model architecture through ablations, shown in Table 3. When either the sinusoidal encodings or MLP blocks are removed, the model performs substantially worse in both legality and HPWL. Replacing attention layers with graph convolutions also causes sample quality to plummet, as evidenced by poor legality scores.

| Model          | #Param. | Legality | HPWL Ratio |
| -------------- | ------- | -------- | ---------- |
| Small          | 0.233M  | 0.948    | 1.072      |
| Medium         | 1.23M   | 0.960    | 1.039      |
| – No attention | 1.42M   | 0.799    | 1.035      |
| – No MLP       | 0.698M  | 0.946    | 1.060      |
| – No encodings | 1.21M   | 0.949    | 1.061      |
| Large          | 6.29M   | 0.976    | 1.032      |


Our model also exhibits favorable scaling properties, with Table 3 showing significant and monotonic improvements in model performance (both legality and HPWL) with increasing model size. This suggests scaling up models as an attractive strategy for improving performance on more complex datasets, particularly since synthetic data is unlimited.

## 5.3. Guided Sampling on Real Circuits

To determine the effectiveness of guidance in improving sample quality, we used our model to generate placements for the IBM benchmark with standard cells clustered.

| Model             | Legality | HPWL (10) |
| ----------------- | -------- | --------- |
| DREAMPlace        | -        | 3.724     |
| Large+v0          | 0.7794   | 3.252     |
| Large+v1          | 0.8213   | 3.281     |
| Large+v2          | 0.8835   | 3.203     |
| Large+v2 (Guided) | 0.9970   | 2.976     |


As shown in Table 4, guidance dramatically improves legality and HPWL during zero-shot sampling, with legality increasing to nearly 1 while simultaneously shortening HPWL by 7.1%. This result shows that our guidance method is effective in optimizing generated samples without requiring additional training. An example of the generated placements with and without guidance is shown in Figure 8.

Moreover, we find that our placements are significantly better than those produced by the state-of-the-art DREAM-Place (Lin et al., 2019), with 20% lower HPWL on average. While we note that DREAMPlace, a mixed-size placer, is not optimized for placing clustered circuits, this result is nevertheless a strong indication of our method's ability to generate high-quality placements for real circuits. This is especially remarkable, showing that models trained entirely on synthetic data can transfer effectively to real circuits in a zero-shot manner.

## 5.4. Mixed-Size Placement of Real Circuits

While our results (Table 4) have shown that our method can produce high-quality macro placements for clustered circuits, we also wish to investigate if these macro placements are useful for downstream tasks, particularly for mixed-size placement.

Figure 9. Our model can be used for mixed-size placement by first placing macros and clusters, then using our placements as inputs to a standard cell placer. ibm03 is shown here.

To perform mixed-size placement (which requires placing all standard cells and macros) using our method, we first use our diffusion models to place clusters and macros (Figure 9a). We then initialize the position of each standard cell to the position of the corresponding cluster, and copy the positions of the macros (Figure 9b). Finally, we use DREAM-Place 4.1 (Lin et al., 2019) to place the standard cells, thus obtaining a full placement of standard cells and macros (Fig-
---


# Chip Placement with Diffusion Models

Table 5. Comparison of average HPWL (10⁶) over 5 seeds using various techniques for mixed-size placement on the IBM benchmark.

| Circuit | MaskPlace + DP | WireMask-BBO + DP | ChiPFormer + DP | DREAMPlace | Diffusion (Ours) |
| ------- | -------------- | ----------------- | --------------- | ---------- | ---------------- |
| ibm01   | 3.33           | 2.84              | 3.35            | 2.23       | 2.09             |
| ibm02   | 7.30           | 6.87              | 6.24            | 5.79       | 4.43             |
| ibm03   | 10.1           | 9.81              | 10.9            | 10.4       | 7.30             |
| ibm04   | 10.4           | 9.65              | 10.1            | 9.13       | 8.00             |
| ibm05   | 7.67           | 7.67              | 7.67            | 7.60       | 7.79             |
| ibm06   | 7.62           | 8.41              | 7.76            | 6.15       | 8.31             |
| ibm07   | 13.3           | 13.0              | 13.4            | 11.1       | 9.60             |
| ibm08   | 15.5           | 15.9              | 15.7            | 12.3       | 13.3             |
| ibm09   | 16.2           | 15.4              | 16.9            | 12.8       | 12.6             |
| ibm10   | 46.8           | 45.2              | 45.4            | 44.8       | 30.2             |
| ibm11   | 23.5           | 24.6              | 23.6            | 16.6       | 17.3             |
| ibm12   | 46.1           | Failed            | 48.8            | 31.0       | 34.0             |
| ibm13   | 28.2           | 28.0              | 28.4            | 23.2       | 23.0             |
| ibm14   | 45.4           | 48.2              | 46.5            | 31.3       | 34.5             |
| ibm15   | 53.4           | Failed            | 55.8            | 51.3       | 45.0             |
| ibm16   | 65.9           | 63.2              | 67.3            | 53.0       | 52.7             |
| ibm17   | 72.9           | 69.7              | 71.4            | 57.9       | 60.4             |
| ibm18   | 42.2           | 41.6              | 41.1            | 37.6       | 38.6             |
| Average | 28.7           | 27.0              | 28.9            | 23.6       | 22.7             |


ure 9c). To elucidate the impact of our macro placements on the final placement quality, we keep the macro positions fixed during standard cell placement. This is in line with earlier works (Lai et al., 2022; Shi et al., 2023; Mirhoseini et al., 2020; Cheng & Yan, 2021) but contrasts with Lai et al. (2023), where the contribution of the macro placement technique is not as clear because the macros move significantly during standard cell placement.

In Table 5, we compare results from our method to other macro placement baselines, as well as DREAMPlace. The baselines include both the learning-based ChiPFormer (Lai et al., 2023) and MaskPlace (Lai et al., 2022), as well as the learning-free WireMask-BBO² (Shi et al., 2023). All of these baselines use DREAMPlace for standard cell placement. We find that our method outperforms prior macro placement methods by a wide margin, while improving over DREAMPlace by 4% on average. This indicates that the strong performance of our model on clustered macro placement transfers well to mixed-size placements. Moreover, our method, by design, can be easily applied zero-shot to new circuits and takes minutes to run, while other methods require RL fine-tuning or black box optimization, spending hours on each new circuit.

²WireMask-BBO fails to place ibm12 and ibm15 so a lower-bound average is computed by substituting the smallest HPWL in the corresponding rows

## 6. Conclusion

In this work, we explored an approach that departs from many existing methods for tackling macro placement: using diffusion models to generate placements. To train and apply such models at scale, we developed a novel data generation algorithm, designed synthetic datasets that enable zero-shot transfer to real circuits, and designed a neural network architecture that performs and scales well. We show that when trained on our synthetic data, our models generalize to new circuits, and when combined with guided sampling, can generate optimized placements even on large, real-world circuit benchmarks.

Even so, our work is not without limitations. RL methods, while slow, provide a means of trading test-time compute for better sample quality. We believe applying such methods, through DDPO (Black et al., 2024) for instance, could combine the strengths of generative modeling and RL fine-tuning. We also note that our synthetic data does not capture all the nuances of real data, such as multimodal edge distributions, and believe this is an interesting area for further study.

In conclusion, we find that training diffusion models on synthetic data is a promising approach, with our models generating competitive placements despite never having trained on realistic circuit data. We hope that our results inspire further work in this area.
---


# Chip Placement with Diffusion Models

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

This work was supported in part by an ONR DURIP grant and the BAIR Industrial Consortium. Pieter Abbeel holds concurrent appointments as a Professor at UC Berkeley and as an Amazon Scholar. This paper describes work performed at UC Berkeley and is not associated with Amazon.

## References

Adya, S., Chaturvedi, S., Roy, J., Papa, D., and Markov, I. Unification of partitioning, placement and floorplanning. In IEEE/ACM International Conference on Computer Aided Design, 2004. ICCAD-2004., pp. 550–557, 2004. doi: 10.1109/ICCAD.2004.1382639.

Bansal, A., Chu, H.-M., Schwarzschild, A., Sengupta, S., Goldblum, M., Geiping, J., and Goldstein, T. Universal guidance for diffusion models, 2023.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training diffusion models with reinforcement learning, 2024.

Brody, S., Alon, U., and Yahav, E. How attentive are graph attention networks?, 2022.

Chen, T.-c., Jiang, Z.-w., Hsu, T.-c., Chen, H.-c., and Chang, Y.-w. A high-quality mixed-size analytical placer considering preplaced blocks and density constraints. In 2006 IEEE/ACM International Conference on Computer Aided Design, pp. 187–192, 2006. doi: 10.1109/ICCAD.2006.320084.

Chen, Y., Mai, J., Gao, X., Zhang, M., and Lin, Y. Macro-rank: Ranking macro placement solutions leveraging translation equivariancy. In Proceedings of the 28th Asia and South Pacific Design Automation Conference, ASP-DAC '23, pp. 258–263, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450397834. doi: 10.1145/3566097.3567899. URL https://doi.org/10.1145/3566097.3567899.

Cheng, C.-K., Kahng, A. B., Kundu, S., Wang, Y., and Wang, Z. Assessment of reinforcement learning for macro placement. In Proceedings of the 2023 International Symposium on Physical Design, ISPD '23, pp. 158–166, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399784.

Cheng, R. and Yan, J. On joint learning for solving placement and routing in chip design, 2021.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021. URL https://arxiv.org/abs/2105.05233.

Fey, M. and Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

Ghose, A., Zhang, V., Zhang, Y., Li, D., Liu, W., and Coates, M. Generalizable cross-graph embedding for gnn-based congestion prediction. In 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1–9. IEEE, 2021.

Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry, 2017.

Gu, H., Gu, J., Peng, K., Zhu, Z., Xu, N., Geng, X., and Yang, J. Lamplace: Legalization-aided reinforcement learning based macro placement for mixed-size designs with preplaced blocks. IEEE Transactions on Circuits and Systems II: Express Briefs, pp. 1–1, 2024. doi: 10.1109/TCSII.2024.3375068.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models, 2020.

Kahng, A. B. and Reda, S. A tale of two nets: Studies of wirelength progression in physical design. In Proceedings of the 2006 international workshop on System-level interconnect prediction, pp. 17–24, 2006.

Karypis, G., Aggarwal, R., Kumar, V., and Shekhar, S. Multilevel hypergraph partitioning: application in vlsi domain. In Proceedings of the 34th Annual Design Automation Conference, DAC '97, pp. 526–529, New York, NY, USA, 1997. Association for Computing Machinery. ISBN 0897919203. doi: 10.1145/266021.266273. URL https://doi.org/10.1145/266021.266273.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes, 2022.

Lai, Y., Mu, Y., and Luo, P. Maskplace: Fast chip placement via reinforced visual representation learning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 24019–24030. Curran Associates, Inc.,
---


# Chip Placement with Diffusion Models

2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/97c8a8eb0e5231d107d0da51b79e09cb-Paper-Conference.pdf.

Lai, Y., Liu, J., Tang, Z., Wang, B., Hao, J., and Luo, P. Chipformer: Transferable chip placement via offline decision transformer. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 18346–18364. PMLR, 2023. URL https://proceedings.mlr.press/v202/lai23c.html.

Lanzerotti, M. Y., Fiorenza, G., and Rand, R. A. Microminiature packaging and integrated circuitry: The work of e. f. rent, with an application to on-chip interconnection requirements. IBM Journal of Research and Development, 49(4.5):777–803, 2005. doi: 10.1147/rd.494.0777.

Lin, Y., Dhar, S., Li, W., Ren, H., Khailany, B., and Pan, D. Z. Dreampiace: Deep learning toolkit-enabled gpu acceleration for modern vlsi placement. In 2019 56th ACM/IEEE Design Automation Conference (DAC), pp. 1–6, 2019.

Liu, Y., Ju, Z., Li, Z., Dong, M., Zhou, H., Wang, J., Yang, F., Zeng, X., and Shang, L. Floorplanning with graph attention. In Proceedings of the 59th ACM/IEEE Design Automation Conference, DAC '22, pp. 1303–1308, New York, NY, USA, 2022a. Association for Computing Machinery. ISBN 9781450391429. doi: 10.1145/3489517.3530484. URL https://doi.org/10.1145/3489517.3530484.

Liu, Y., Ju, Z., Li, Z., Dong, M., Zhou, H., Wang, J., Yang, F., Zeng, X., and Shang, L. Graphplanner: Floorplanning with graph neural network. ACM Trans. Des. Autom. Electron. Syst., 28(2), dec 2022b. ISSN 1084-4309. doi: 10.1145/3555804. URL https://doi.org/10.1145/3555804.

Mirhoseini, A., Goldie, A., Yazgan, M., Jiang, J. W. J., Songhori, E. M., Wang, S., Lee, Y., Johnson, E., Pathak, O., Bae, S., Nazi, A., Pak, J., Tong, A., Srinivasa, K., Hang, W., Tuncer, E., Babu, A., Le, Q. V., Laudon, J., Ho, R., Carpenter, R., and Dean, J. Chip placement with deep reinforcement learning. CoRR, abs/2004.10746, 2020. URL https://arxiv.org/abs/2004.10746.

Mirhoseini, A., Goldie, A., Yazgan, M., Jiang, J. W., Songhori, E., Wang, S., Lee, Y.-J., Johnson, E., Pathak, O., Nazi, A., et al. A graph placement methodology for fast chip design. Nature, 594(7862):207–212, 2021.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,

L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Shi, Y., Xue, K., Song, L., and Qian, C. Macro placement by wire-mask-guided black-box optimization, 2023. URL https://arxiv.org/abs/2306.16844.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations, 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, B., Shen, G., Li, D., Hao, J., Liu, W., Huang, Y., Wu, H., Lin, Y., Chen, G., and Heng, P. A. Lhnn: Lattice hypergraph neural network for vlsi congestion prediction. In Proceedings of the 59th ACM/IEEE Design Automation Conference, pp. 1297–1302, 2022.

Yue, S., Songhori, E. M., Jiang, J. W., Boyd, T., Goldie, A., Mirhoseini, A., and Guadarrama, S. Scalability and generalization of circuit training for chip floorplanning. In Proceedings of the 2022 International Symposium on Physical Design, ISPD '22, pp. 65–70, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392105. doi: 10.1145/3505170.3511478. URL https://doi.org/10.1145/3505170.3511478.

Zheng, S., Zou, L., Liu, S., Lin, Y., Yu, B., and Wong, M. Mitigating distribution shift for congestion optimization in global placement. In 2023 60th ACM/IEEE Design Automation Conference (DAC), pp. 1–6, 2023a. doi: 10.1109/DAC56929.2023.10247660.

Zheng, S., Zou, L., Xu, P., Liu, S., Yu, B., and Wong, M. Lay-net: Grafting netlist knowledge on layout-based congestion prediction. In 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD), pp. 1–9. IEEE, 2023b.
---


# Chip Placement with Diffusion Models

## A. Implementation Details

Hyperparameters for our model are listed in Table 6. We used the DDPM formulation found in Ho et al. (2020) with a cosine noise schedule over 1000 steps. We also list parameters for generating our synthetic datasets in Table 7.

|                                      | Small                      | Medium | Large |
| ------------------------------------ | -------------------------- | ------ | ----- |
| **Model Dimensions**                 |                            |        |       |
| Model size                           | 64                         | 128    | 256   |
| Blocks                               | 2                          | 2      | 3     |
| Layers per block                     | 2                          |        |       |
| AttGNN size                          | 32                         | 32     | 256   |
| ResGNN size                          | 64                         | 256    | 256   |
| MLP size factor                      | 4                          |        |       |
| MLP layers per block                 | 2                          |        |       |
| **Input Encodings**                  |                            |        |       |
| Timestep encoding dimension          | 32                         |        |       |
| Input encoding dimension             | 32                         |        |       |
| **GNN Layers**                       |                            |        |       |
| Type                                 | GATv2 (Brody et al., 2022) |        |       |
| Heads                                | 4                          |        |       |
| **Guidance Parameters**              |                            |        |       |
| wHPWL                                | 0.0001                     |        |       |
| $\hat{x}\_0$ optimizer               | SGD                        |        |       |
| $\hat{x}\_0$ optimizer learning rate | 0.008                      |        |       |
| wlegality optimizer                  | Adam                       |        |       |
| wlegality optimizer learning rate    | 0.0005                     |        |       |
| wlegality initial value              | 0                          |        |       |
| Gradient descent steps               | 10                         |        |       |
| ε                                    | 0.0001                     |        |       |

---


# Chip Placement with Diffusion Models

Table 7. Parameters for generating our synthetic data. Unless otherwise stated, v1 and v2 use the same parameters as v0. We refer the reader to our code for more details. Sizes and distances are normalized so that a value of 2 corresponds to the width of the canvas.

|                               | v0                  | v1                 | v2                  |
| ----------------------------- | ------------------- | ------------------ | ------------------- |
| **Stop Density Distribution** |                     |                    |                     |
| Distribution type             | Uniform             |                    |                     |
| Low                           | 0.75                |                    |                     |
| High                          | 0.9                 |                    |                     |
| **Aspect Ratio Distribution** |                     |                    |                     |
| Distribution type             | Uniform             |                    |                     |
| Low                           | 0.25                |                    |                     |
| High                          | 1.0                 |                    |                     |
| **Object Size Distribution**  |                     |                    |                     |
| Distribution type             | Clipped Exponential |                    |                     |
| Scale                         | 0.08                |                    | 0.04                |
| Max                           | 1.0                 |                    | 0.5                 |
| Min                           | 0.02                |                    | 0.01                |
| **Edge Distribution**         |                     |                    |                     |
| Distribution p(l)             | γ · exp (−l/s)      |                    |                     |
| Scale s                       | 0.2                 | ∼ log U(0.05, 1.6) | ∼ log U(0.025, 0.8) |
| Max p                         | 0.9                 |                    |                     |
| γ                             | 0.21                | 0.212 · s−1.42     | 0.00792 · s−1.42    |
