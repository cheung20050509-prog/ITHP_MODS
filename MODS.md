## Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection

### Dingkang Yang[1,2], Mingcheng Li[1][∗], Xuecheng Wu[3], Zhaoyu Chen[1], Kaixun Jiang[1], Keliang Liu[1], Peng Zhai[1], Lihua Zhang[1*]

1College of Intelligent Robotics and Advanced Manufacturing, Fudan University
2ByteDance
3School of Computer Science and Technology, Xi’an Jiaotong University
yangdingkang@bytedance.com, lihuazhang@fudan.edu.cn


**Abstract**

Multimodal Sentiment Analysis (MSA) aims to predict sentiment from language, acoustic, and visual data in videos.
However, imbalanced unimodal performance often leads to
suboptimal fused representations. Existing approaches typically adopt fixed primary modality strategies to maximize
dominant modality advantages, yet fail to adapt to dynamic
variations in modality importance across different samples.
Moreover, non-language modalities suffer from sequential
redundancy and noise, degrading model performance when
they serve as primary inputs. To address these issues, this
paper proposes a modality optimization and dynamic primary modality selection framework (MODS). First, a Graphbased Dynamic Sequence Compressor (GDC) is constructed,
which employs capsule networks and graph convolution to
reduce sequential redundancy in acoustic/visual modalities.
Then, we develop a sample-adaptive Primary Modality Selector (MSelector) for dynamic dominance determination. Finally, a Primary-modality-Centric Cross-Attention (PCCA)
module is designed to enhance dominant modalities while
facilitating cross-modal interaction. Extensive experiments
on four benchmark datasets demonstrate that MODS outperforms state-of-the-art methods, achieving superior performance by effectively balancing modality contributions and
eliminating redundant noise.


### Introduction

In the modern digital era, individuals frequently share opinions and emotions through social media and e-commerce
platforms. Sentiment analysis of such data holds broad application value (Gallagher, Furey, and Curran 2019; Tsai
and Wang 2021; Drus and Khalid 2019; Yang et al. 2022c,
2023a). Traditional unimodal sentiment analysis (USA)
methods (Ortis, Farinella, and Battiato 2019; Xiao, Yang,
and Ning 2021) rely solely on a single data source (e.g., language or visual) and suffer from inherent limitations such as
informational ambiguity and poor noise resistance. Emerging multimodal sentiment analysis (MSA) techniques enhance analytical accuracy by integrating multidimensional
information from different modalities, better aligning with
natural human emotional expression, where emotions are of
*Corresponding authors.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.


ten conveyed through multiple channels, such as language
content, facial expressions, and vocal tones.
In the MSA task (Mai et al. 2022), different modalities contribute unevenly to sentiment prediction. The language modality, due to its condensed emotional expression
and high semantic density, is typically regarded as the primary information source, particularly excelling in opinionoriented or conversational scenarios. Consequently, prior
studies have developed a series of language-oriented MSA
methods (Wu et al. 2021; Lei et al. 2023; Zhang et al. 2023)
to use the advantages of language data. While these methods recognize intermodal disparities and improve performance, their static modality-dominant strategies have limitations: they capture population-level patterns but fail to
adapt to dynamic sample-wise variations in modality dominance. Specifically, when non-language modalities dominate emotional expression in certain samples, forcibly prioritizing the language modality causes models to overlook
critical affective cues from other modalities, thereby compromising prediction performance. Previous work (Wang
et al. 2023b) identifies this issue and proposes the HCTDMG for dynamic primary modality selection. However,
HCT-DMG overlooks another key problem: the inherent sequential redundancy of non-language modalities. Compared
to language modality, serialized representations of acoustic
and visual modalities exhibit significantly lower information density, containing more repetitive and irrelevant features. Directly treating them as primary modalities may introduce noise interference, degrading fusion performance.
Furthermore, asynchronous multimodal sequences restrict
HCT-DMG to batch-level (rather than sample-level) primary
modality selection, preventing true sample-wise adaptation.
To address these challenges, we propose a modality
optimization and dynamic primary modality selection
framework (MODS), a new MSA algorithm supporting
sample-level dynamic modality selection. Specifically, we
first design a Graph-based Dynamic Compressor (GDC)
module to resolve sequential redundancy in non-language
modalities via graph convolution operations. This module employs capsule networks (Sabour, Frosst, and Hinton
2017) for efficient graph structure modeling, compressing
redundant information while eliminating noise in acousticvisual modalities to enhance feature quality and emotional
expressiveness. Then, we present a sample-adaptive MS

-----

elector module to transcend conventional unimodal dominance paradigms by dynamically determining the optimal
primary modality for each input sample. Automatic modality selection based on sample characteristics enables flexible
cross-scenario adaptability. Finally, we introduce a Primarymodality-Centric Crossmodal Attention (PCCA) module to
facilitate intermodal interaction and primary modality enhancement. These components improve MSA performance
by simultaneously capturing dynamic modality dominance
patterns and eliminating redundant interference, delivering
a more robust and generalizable solution for complex multimodal tasks. We develop the MODS and conduct comprehensive experiments on four public MSA datasets. Empirical
results and analyses validate the framework’s effectiveness.

### Related Work
**Ternary Symmetric-based MSA. Benefiting from deep**
learning technologies (Wang et al. 2025b), MSA methods primarily focus on multimodal representation learning and multimodal fusion (Zadeh et al. 2018a; Hazarika,
Zimmermann, and Poria 2020; Tsai et al. 2019; Yang
et al. 2022b,a,d, 2023b). Mainstream representation learning and fusion approaches typically treat different modalities
equally. For instance, Zadeh et al. (Zadeh et al. 2018a) proposed the Memory Fusion Network (MFN) that combines
LSTM with attention mechanisms, utilizing their Deltamemory attention module to achieve cross-modal interaction at each timestep of aligned multimodal data. Hazarika et al. (Hazarika, Zimmermann, and Poria 2020) introduced MISA, projecting each modality into two distinct
subspaces to separately learn modality-invariant shared features and modality-specific characteristics. Tsai et al. (Tsai
et al. 2019) created the MulT that employs pairwise crossmodal attention to align and fuse asynchronous multimodal
sequences without manual alignment. Building upon MulT,
Lv et al. (Lv et al. 2021) proposed the PMR model, which
extends bidirectional cross-modal attention to tri-directional
attention involving all modalities while introducing a message hub to explore intrinsic inter-modal correlations for
more efficient multimodal feature fusion. Liang et al. (Liang
et al. 2021) developed the MICA architecture based on
MulT, incorporating modality distribution alignment strategies into cross-modal attention operations to obtain reliable
cross-modal representations for asynchronous multimodal
sequences. Mai et al. (Mai et al. 2022) presented HyCon using hybrid contrastive learning for trimodal representation,
employing three distinct contrastive learning models to capture inter-modal interactions and inter-class relationships after obtaining unimodal representations.
**Language Center-based MSA. Another prominent line**
of research focuses on enhancing the weighting of language modality in MSA tasks to better leverage dominant
modalities, aiming to overcome performance bottlenecks
in MSA. For instance, existing efforts (Delbrouck et al.
2020; Han et al. 2021; Wang et al. 2023a, 2025a) employed
transformer-based approaches to integrate complementary
information from other modalities through language modality. Han et al. (Han, Chen, and Poria 2021) further improved
this by maximizing mutual information between non-textual


and textual modalities to filter out task-irrelevant modalityspecific noise. Meanwhile, some works (Li et al. 2022; Lin
and Hu 2022) introduced contrastive learning to capture
shared features between non-language and language modalities. Furthermore, Wu et al. (Wu et al. 2021) proposed
a Text-Centric Shared-Private (TCSP) framework that distinguishes between shared and private semantics of textual
and non-textual modalities, effectively fusing text features
with two types of non-text features. Building upon Transformer (Vaswani et al. 2017) architectures, impressive methods (Lei et al. 2023; Zhang et al. 2023; Wu et al. 2023) have
been designed that establish language modality’s dominant
role to achieve sequence fusion. Although the aforementioned types of MSA methods have achieved encouraging
results, performance bottlenecks still exist. Approaches that
treat each modality equally may cause the model to be distracted by secondary modalities, while methods focused on
the language modality lack flexibility and struggle to adapt
to variations in dominant modality across samples.

### Methodology
#### Problem Statement
The MSA task focuses on detecting sentiments by leveraging the language (l), visual (v), and acoustic (a) modalities
from the video clips. Given unimodal sequences from these
modalities as Xm ∈ R[T][m][×][d][m], where m ∈{l, v, a}. Tm and
_dm are used to represent the sequence length and the feature_
dimension of the modality m, respectively.

#### Framework Overview
The overall architecture of the MODS algorithm is illustrated in Figure 1. First, preliminary feature extraction is
performed on each modality using corresponding methods
to obtain unimodal sequence features Hm, where m ∈
_l, a, v_ . For the language features, after extraction via
_{_ _}_
BERT, a linear layer projects the feature dimension to d,
yielding Hl ∈ R[T][l][×][d]. Meanwhile, the acoustic and visual
features are compressed to the same dimension as the language features through a graph convolutional sequence compression module, resulting in Ha ∈ R[T][l][×][d] and Hv ∈ R[T][l][×][d],
respectively. The three unimodal features are then fed into
the Primary Modality Selector (MSelector) to determine the
primary modality p and secondary modalities a1, a2 along
with their weighted sequence feature outputs Hp, Ha1, and
_Ha2. After primary modality selection, the three modal-_
ity features are progressively processed by the Primarymodality-Centric Crossmodal Attention (PCCA) module for
multimodal interaction and fusion. The final output is the enhanced fused feature Hp. Ultimately, Hp undergoes adaptive
aggregation to produce a one-dimensional vector hp, which
is then passed through a multilayer perceptron (MLP) to obtain the sentiment prediction result ypred.

#### Graph-based Dynamic Compression
RNNs (Hochreiter and Schmidhuber 1997) were widely
used to study sequential semantics in feature embedding
modeling. However, it suffers from slow training, difficulty
in capturing long-range dependencies due to their recurrent


-----

Figure 1: The overall architecture of the proposed MODS framework.


nature, and gradient-related issues. Transformers (Vaswani
et al. 2017) enable parallel computation and improve longrange dependency modeling, but rely on attention-weighted
summation, making explicit temporal relationship modeling challenging. Both lack sequence compression capability: RNNs process data step-by-step inefficiently, while
Transformers incur redundant computations and suboptimal
weight allocation. In contrast, graph structures enhance information efficiency via direct node connections, and graph
convolutional networks (GCNs) (Kipf and Welling 2016)
improve stability through neighbourhood aggregation, mitigating gradient problems. Thus, we propose the Graphbased Dynamic Compression (GDC) module to project nonlanguage sequences into graph space, exploring intra-modal
dependencies while compressing redundant long sequences.
In prior sequence modeling frameworks, graph node construction generally follows one of two strategies: treating
features at each timestep as individual nodes, which preserves sequence fidelity at the cost of computational redundancy, or employing temporal slice pooling, which improves
efficiency but may compromise fine-grained temporal information. Instead of these, we propose a capsule networkbased solution, where its dynamic routing mechanism adaptively learns node importance weights, and vectorized feature representation preserves directional and semantic information. This enables sequence compression while avoiding
the loss of critical details. The detail of the GDC module
is shown in Figure 2. We first compress the sequence information into an appropriate number of nodes using a capsule network. To facilitate model training and subsequent
cross-modal interactions, we align the compressed sequence
lengths of acoustic and visual modalities with that of the language modality. Specifically, for the sequence features Hm
of modality m _a, v, the capsule Caps construction pro-_
_∈_
cess is defined as follows:

_Caps[i,j]m_ [=][ W]m[ ij][H]m[i] _[,]_ (1)


where Hm[i] _[∈]_ _[R][d][m]_ [denotes the feature at the][ i][-th timestep of]
sequence Hm, Wm[ij] _[∈]_ _[R][d][×][d][m][ represents the trainable pa-]_
rameters, and Caps[i,j]m _[∈]_ _[R][d][ is the capsule created from the]_
_i-th timestep feature for constructing the j-th node. Subse-_
quently, the graph node representations are generated based
on the defined capsules and the dynamic routing algorithm:

_Nm[j]_ [=] � _Caps[i,j]m_ _m_ _[,]_ (2)

_[×][ r][i,j]_
_i_

where rm[i,j] [is the is the routing coefficient, which is updated]
through an iterative optimization process:

exp(b[i,j]m [)]
_rm[i,j]_ [=] _,_ (3)
�j [exp(][b]m[i,j][)]

_b[i,j]m_ _m_ [+][ Caps]i,j[m] _m[)][.]_ (4)

_[←]_ _[b][i,j]_ _[⊙]_ _[tanh][(][N][ j]_

Here, b[i,j]m [is the unnormalized routing coefficient initial-]
ized to 0. After normalization, the initial value of rm[i,j] [is]
1
_n_ [(where][ n][ is the number of capsules), meaning each cap-]
sule is initially assigned equal routing coefficients. According to the properties of capsule networks, during the iterative process of dynamic routing, capsules receiving noise
or redundant information will be gradually assigned smaller
weights, while capsules containing important affective information will obtain larger weights. Therefore, using capsule
networks to create graph nodes helps improve the quality
of acoustic-visual features and increase the density of effective information in sequences. After obtaining the node
representations of the modality feature graph, we employ a
self-attention mechanism to explore the edge weight relationships between nodes in the feature graph:


�


_Em = ReLU_


� (Wm[q] _[N][m][)][T][ �]Wm[k]_ _[N][m]�_

_√_

_d_


_._ (5)


-----

Figure 2: The architecture of the proposed GDC module.

The edge construction process, empowered by the selfattention mechanism, focuses on the most relevant information in the modality feature graph to generate edge representations. This process effectively strengthens the dependencies between nodes, ensuring that the generated edge representations can more accurately reflect the inter-node relationships within the graph.
GCNs are pivotal for graph-based learning. GCNs enable
simultaneous processing of complex data, capture essential
global information, and demonstrate high efficacy in learning both nodes and edges. Therefore, for the modality feature graph Gm = (Nm, Em), we employ GCN to perform
representation learning on the graph. The specific formulation is as follows:

� _−_ [1]2 _−_ [1]2 �
_Hm[l]_ [=][ ReLU] _Dm_ _[E]m[D]m_ _[H]m[l][−][1][W][ l]m_ [+][ b]m[l] _,_ (6)


where Hm[l] [is the output of][ l][-th GCN,][ m][ ∈{][l, a, v][}][,][D][m]
is the degree matrix of Em, Wm[l] [and][ b]m[l] [are the learnable]
parameters of the l-th GCN layer. The initial input to the
GCN is the set of node representations from the modality
feature graph, denoted as Hm[0] [=][ N][ j]m[.]

#### Primary Modality Selection

The adaptive dynamic primary modality selection module
(MSelector) aims to enable the network to autonomously select the primary modality through training, eliminating the
need for manual intervention. The core mechanism of this
module involves dynamically evaluating the importance of
each modality by learning its trainable parameters. First, it
performs adaptive aggregation on each modality’s sequential features, transforming them into one-dimensional feature vectors. Specifically, for Hm ∈ _R[T][m][×][d], the attention_
weight matrix is first computed as follows:


with the sequential features to obtain the aggregated vector:

_hm = amHm_ _R[1][×][d]._ (8)
_∈_

Subsequently, the vectors from the three modalities are concatenated and fed into a Multilayer Perceptron (MLP) for
processing:

_w = softmax(MLP_ (concat(ha, hl, hv))). (9)

The output of the MLP is a three-dimensional vector, which
is processed through a softmax function to generate three
weight values that sum to 1:

_w = [wa, wt, wv],_ _wa + wt + wv = 1._ (10)

Each weight value represents the importance degree of a corresponding modality, where a higher weight indicates that
the model considers the modality to have greater contribution. The modality with the highest weight is identified as
the primary modality p:

_p = arg max(wa, wt, wv)._ (11)

To further enhance the influence of the primary modality
while distinguishing between the suboptimal modality a1
and a2, we multiply these weight values with their corresponding modality features, and use the resulting products
as inputs to the subsequent network:

_Ha1 = wa · Ha,_ _Ha2 = wt · Ht,_ _Hp = wv · Hv. (12)_

It should be noted that the formula here is written with
modality v as the primary modality for illustration purposes.
During training and inference, the primary modality dynamically varies based on the model’s selection.

#### Primary-modality-Centric Interaction
To capture element correlations among multimodal sequences and enhance the primary modality, we design the
PCCA module to enable cross-modal interactions. In the
PCCA module, a two-step process is employed to facilitate information flow among three modalities. Initially, information from the suboptimal modality flows to the primary modality via the CA[[]a[i]→[]] _p_ [block and is then fused.]
Subsequently, the fused information flows from the primary
modality to the suboptimal modality via the CA[[]p[i]→[]] _a_ [block.]
This module ensures a mutual information flow and progressive enhancement across all modalities. The architecture of PCCA[[][i][]] is shown in Figure 3, and the superscript

[i] indicates the i-th modality reinforcement processes. Its
inputs are Hp[[][i][]][,][ H]a[[][i]1[]][, and][ H]a[[][i]2[]] [while its outputs are the re-]
inforced features of these three modalities as Hp[[][i][+1]], Ha[[][i]1[+1]]
and Ha[[][i]2[+1]]:

�
_Hp[[][i][+1]], Ha[[][i]1[+1]], Ha[[][i]2[+1]]_ = PCCA[[][i][]][ �]Hp[[][i][]][, H]a[[][i]1[]][, H]a[[][i]2[]] _._
(13)
Specifically, we first perform a layer normalization (LN ) on
the features Hm[[][i][]][, where][ H]m[[0]] [=][ H]m [and][ m][ ∈{][p, a][1][, a][2][}][.]
Then, two cross-attention and one self-attention are simultaneously implemented. The cross-attention block CA[[]a[i]→[]] _l_


� _HmWm_
_am = softmax_ _√_

_d_


�T
_R[1][×][T][m]_ _,_ (7)
_∈_


where m ∈{l, a, v}. Wm ∈ _R[d]_ is the linear projection
parameters and am represents the attention weight matrix
for Hm. Subsequently, the attention weights are multiplied


-----

takes Hp[[][i][]] [as Quary and][ H]a[[][i][]] [as Key and Value, and obtain]
_Ha[[][i]→[]]_ _p_ [with][ a][ ∈{][a][1][, a][2][}][ :]

� �
_Ha[[][i]→[]]_ _p_ [=][ CA]a[[][i]→[]] _p_ _Ha[[][i][]][, H]p[[][i][]]_ _._ (14)

Meanwhile, the self-attention block SA[[]p[i][]] [takes][ H]p[[][i][]] [as input]

to obtain Hp[[][i]update[+1]] [:]

� �
_Hp[[][i]update[]]_ [=][ SA]p[[][i][]] _Hp[[][i][]]_ + Hp[[][i][]][.] (15)

Here, SA[[]p[i][]] aims to capture the contextual relationships inside the modality and realize the modality’s selfreinforcement. Now we obtain primary modality features
that capture both intra-modal and inter-modal dependencies,
and then we add them together to get a complete enhancement of the primary modality:

�
_Hp[[][i][+1]]_ = Hp[[][i]update[]] [+] _Ha[[][i]→[]]_ _p[.]_ (16)

_a∈{a1,a2}_

Since Hp[[][i][+1]] contains information from all modalities, we
then perform two cross-attention operations CA[[]p[i]→[]] _a_ [to pass]
the information of the enhanced language modality to the
visual/acoustic modalities, which is expressed as follows:

� �
_Hp[[][i]→[]]_ _a_ [=][ CA]p[[][i]→[]] _a_ _Hp[[][i][+1]], Ha[[][i][]]_ _._ (17)

In doing so, we utilize the primary modality as a bridge
to implicitly realize the information flow among all three
modalities. Subsequently, we process Hp[[][i][+1]] and Hp[[][i]→[]] _a_ [by]
a Position-wise Feed-Forward layer (PFF ) with skip connection:

� � ��
_Ha[[][i][+1]]_ = PFF _LN_ _Hp[[][i]→[]]_ _a_ + Hp[[][i]→[]] _a[,]_ (18)

� � ��
_Hp[[][i][+1]]_ = PFF _LN_ _Hp[[][i][+1]]_ + Hp[[][i][+1]]. (19)

It should be noted that the final PCCA module retains only
the CAa→p operation, with its enhanced output feature Hp
being used for downstream regression tasks. Since direct
interactions between suboptimal modalities make it easy
to generate interference information, our proposed PCCA
module uses the primary modality as a bridge so that suboptimal modalities’ information can be fused under the primary modality’s feature monitoring, thereby reducing the
generation of unnecessary semantics.

#### Training Objective Optimization
In the early stages of training, the selection of the primary
modality may be influenced by weight initialization or data
distribution, leading to instability in the choice. To address
this, we introduce the InfoNCE loss to enhance the stability of primary modality selection and constrain the model to
retain key information across different modalities. Specifically, for the output Hp of the PCCA module, we first aggregate it into a one-dimensional vector hp using the adaptive aggregation method described above. Then, we construct a reverse prediction path F from the feature hp to the


Figure 3: The architecture of the proposed PCCA module.

unimodal feature hm, and measure the correlation between
them using the following normalized similarity function:

sim(hm, hp) = exp�(hm/∥hm∥[2]) ⊙ (F(hp)/∥F(hp)∥[2])� _,_
(20)
where F takes hp as input and produces a prediction for
_hm, ∥· ∥[2]_ denotes the Euclidean norm normalization, and
represents element-wise multiplication. Then, the above
_⊙_
similarity function is incorporated into the noise-contrastive
estimation framework to generate the InfoNCE loss:

 

exp (sim(h[+]m[,][ F][(][h][p][)))]

_LNCE[f,m]_ [=][ −][E][h]m[,h]p log _K_ � �  _._

�k=1 [exp] sim(h[˜][k]m, F(hp))

(21)
_h˜m = {h˜[1], ˜h[2]...h˜[K]} represents all other samples in the_
same batch during training, where K is the batch size. These
samples are treated as negative samples for contrastive learning. Finally, the InfoNCE loss for all modalities is computed
as follows:

_LNCE = LNCE[p,l]_ [+][ L]NCE[p,a] [+][ L]NCE[p,v] _[.]_ (22)

In the output, we input the representation hp into an MLP
to predict sentiment score ypred. Given the predictions ypred
and the ground truth ytrue, we calculate the task loss Lreg
by mean absolute error (MAE). Finally, we training MODS
by the union loss Ltask:

_ypred = MLP (hp; θΦ),_ (23)


-----

|MOSI Dataset MOSEI Dataset Methods MAE ↓ Corr ↑ Acc7 ↑ Acc2 ↑ F1 ↑ MAE ↓ Corr ↑ Acc7 ↑ Acc2 ↑ F1 ↑|MOSEI Dataset|
|---|---|
||MAE ↓ Corr ↑ Acc7 ↑ Acc2 ↑ F1 ↑|
|TFN (Zadeh et al. 2017)† 0.947 0.673 34.46 77.99/79.08 77.95/79.11 LMF (Liu et al. 2018)† 0.950 0.651 33.82 77.90/79.18 77.80/79.15 ICCN (Sun et al. 2020) 0.862 0.714 39.0 -/83.0 -/83.0 MulT (Tsai et al. 2019)† 0.879 0.702 36.91 79.71/80.98 79.63/80.95 MISA (Hazarika, Zimmermann, and Poria 2020)† 0.776 0.778 41.37 81.84/83.54 81.82/83.58 Self-MM (Yu et al. 2021)† 0.708 0.796 46.67 83.44/85.46 83.36/85.43 MMIM (Han, Chen, and Poria 2021)∗ 0.718 0.797 46.64 83.38/85.82 83.29/85.81 MSG (Lin and Hu 2023) 0.748 0.782 47.3 -/85.6 /85.6 PriSA (Ma, Zhang, and Sun 2023)∗ 0.719 0.782 47.1 83.3/85.4 83.1/85.4 DMD (Li, Wang, and Cui 2023)∗ - - 43.9 - /84.9 - /85.0 MIM (Zeng et al. 2023)∗ 0.718 0.792 46.4 - /84.8 - /84.8 DTN (Zeng et al. 2024)∗ 0.716 0.790 47.5 - /85.1 - /85.1 MODS (Ours) 0.688 0.798 49.27 83.53/85.83 83.75/85.96|0.572 0.714 51.60 78.50/81.89 78.96/81.74 0.576 0.717 51.59 80.54/83.48 80.94/83.36 0.565 0.713 51.6 -/84.2 -/84.2 0.559 0.733 52.84 81.15/84.63 81.56/84.52 0.568 0.724 - 82.59/84.23 82.67/83.97 0.531 0.764 53.87 83.76/85.15 83.82/84.90 0.537 0.759 53.42 82.08/85.14 82.51/85.11 0.583 0.787 52.8 -/85.4 -/85.4 0.536 0.761 53.95 82.1/85.2 83.3/85.2 - - 53.1 - /85.2 - /85.2 0.579 0.779 51.8 - /85.7 - /85.6 0.572 0.765 52.3 - /85.5 - /85.5 0.527 0.772 54.32 84.52/85.88 84.5/86.14|


Table 1: Comparison results on the MOSI and MOSEI datasets. : the results from (Mao et al. 2022); : the results are repro_†_ _∗_
duced from the open-source codebase with hyper-parameters provided in original papers. For Acc2 and F 1, we have two sets
of non-negative/negative (left) and positive/negative (right) evaluation results. Bold represents the best results.

SIMS Dataset SIMSv2 Dataset
Methods

_MAE ↓_ _Corr ↑_ _Acc2 ↑_ _Acc3 ↑_ _Acc5 ↑_ _F_ 1 ↑ _MAE ↓_ _Corr ↑_ _Acc2 ↑_ _Acc3 ↑_ _Acc5 ↑_ _F_ 1 ↑

TFN (Zadeh et al. 2017)[†] 0.432 0.591 78.3 65.12 39.3 78.62 0.329 0.640 77.95 70.21 51.93 77.74

LMF (Liu et al. 2018)[†] 0.441 0.576 77.77 64.68 40.53 77.88 0.367 0.557 74.18 64.90 47.79 73.88

MulT (Tsai et al. 2019)∗ 0.453 0.564 78.56 64.77 37.94 79.66 0.304 0.705 79.3 72.63 53.29 79.43

Self-MM (Yu et al. 2021)[†] 0.425 0.595 80.04 65.47 41.53 80.44 0.322 0.678 79.11 72.34 53.0 79.05

CENet (Wang et al. 2022)[†] 0.471 0.534 77.90 62.58 33.92 77.53 0.310 0.699 79.56 73.10 53.04 **79.63**

ALMT (Zhang et al. 2023)[∗] 0.408 0.594 78.77 65.86 43.11 78.71 0.308 0.700 79.59 71.86 52.90 79.51

DMD (Li, Wang, and Cui 2023)[∗] 0.412 0.586 78.33 65.23 44.26 79.21 0.305 0.702 78.87 72.01 53.18 79.21

MIM (Zeng et al. 2023)[∗] 0.420 0.592 78.98 65.12 44.98 78.70 0.310 0.694 77.56 71.45 52.87 78.56

DTN (Zeng et al. 2024)[∗] 0.419 0.593 79.45 65.67 44.26 79.47 0.302 0.701 78.29 72.56 53.71 78.12

**MODS (Ours)** **0.407** **0.605** **80.96** **66.74** **45.51** **80.94** **0.297** **0.712** **79.59** **73.69** **55.51** 79.53

Table 2: Comparison results on the SIMS and SIMSv2 datasets. : the results from (Mao et al. 2022); : the results are repro_†_ _∗_
duced from the open-source codebase with hyper-parameters provided in original papers. Bold represents the best results.

|SIMS Dataset SIMSv2 Dataset Methods MAE ↓ Corr ↑ Acc2 ↑ Acc3 ↑ Acc5 ↑ F1 ↑ MAE ↓ Corr ↑ Acc2 ↑ Acc3 ↑ Acc5 ↑ F1 ↑|SIMSv2 Dataset|
|---|---|
||MAE ↓ Corr ↑ Acc2 ↑ Acc3 ↑ Acc5 ↑ F1 ↑|
|TFN (Zadeh et al. 2017)† 0.432 0.591 78.3 65.12 39.3 78.62 LMF (Liu et al. 2018)† 0.441 0.576 77.77 64.68 40.53 77.88 MulT (Tsai et al. 2019)∗ 0.453 0.564 78.56 64.77 37.94 79.66 Self-MM (Yu et al. 2021)† 0.425 0.595 80.04 65.47 41.53 80.44 CENet (Wang et al. 2022)† 0.471 0.534 77.90 62.58 33.92 77.53 ALMT (Zhang et al. 2023)∗ 0.408 0.594 78.77 65.86 43.11 78.71 DMD (Li, Wang, and Cui 2023)∗ 0.412 0.586 78.33 65.23 44.26 79.21 MIM (Zeng et al. 2023)∗ 0.420 0.592 78.98 65.12 44.98 78.70 DTN (Zeng et al. 2024)∗ 0.419 0.593 79.45 65.67 44.26 79.47 MODS (Ours) 0.407 0.605 80.96 66.74 45.51 80.94|0.329 0.640 77.95 70.21 51.93 77.74 0.367 0.557 74.18 64.90 47.79 73.88 0.304 0.705 79.3 72.63 53.29 79.43 0.322 0.678 79.11 72.34 53.0 79.05 0.310 0.699 79.56 73.10 53.04 79.63 0.308 0.700 79.59 71.86 52.90 79.51 0.305 0.702 78.87 72.01 53.18 79.21 0.310 0.694 77.56 71.45 52.87 78.56 0.302 0.701 78.29 72.56 53.71 78.12 0.297 0.712 79.59 73.69 55.51 79.53|


and SIMSv2, the accuracy of 7-class (Acc7) on MOSI and
MOSEI, and the accuracy of 2-class (Acc2), Mean Absolute
Error (MAE), Pearson Correlation (Corr), and F1-score
(F 1) on all datasets. In particular, higher values indicate better performance for all metrics except MAE.

#### Implementation Details

Following the (Yu et al. 2021), we use unaligned raw data
in all experiments. All models are built on the Pytorch toolbox (Paszke et al. 2019) with two Quadro RTX 8000 GPUs.
The Adam optimizer (Kingma and Ba 2014) is adopted
for network optimization. The training duration of each
model is governed by an early-stopping strategy with the
patience of 25 epochs for MOSI, SIMS and SIMSv2 and
15 epochs for MOSEI. For MOSI, MOSEI, SIMS, and
SIMSv2, the detailed hyper-parameter settings are as follows: the learning rates are 3e 5, 1e 5, 1e 5, 1e 5,
_{_ _−_ _−_ _−_ _−_ _}_
the batch sizes are 32, 64, 32, 32, the hidden size are
_{_ _}_


_Lreg = [1]_

_N_


_N_
�

_|ytrue −_ _ypred|,_ (24)
_i=1_


_Ltask = Lreg + αLNCE,_ (25)

where α is a parameter that balances the loss contribution.

### Experiments
#### Datasets and Evaluation Metrics
We conduct experiments on four publicly benchmark
datasets of MSA, including MOSI (Zadeh et al. 2016),
MOSEI (Zadeh et al. 2018b), SIMS (Yu et al. 2020), and
SIMSv2 (Liu et al. 2022). In contrast to the MOSI and MOSEI, both SIMS and SIMSv2 prioritize balanced modalityspecific sentiment dominance, avoiding a clear trend where
any single modality dominates emotional expression. This
design better validates the effectiveness of MODS. We use
the accuracy of 3-class (Acc3) and 5-class (Acc5) on SIMS


-----

SIMS Dataset MOSI Dataset
Setting

_MAE ↓_ _Corr ↑_ _Acc5 ↑_ _MAE ↓_ _Corr ↑_ _Acc7 ↑_

**MODS (Full)** **0.407** **0.605** **45.51** **0.688** **0.798** **49.27**

w/o GDC 0.428 0.584 42.01 0.731 0.792 45.34
w/o Caps 0.426 0.582 43.33 0.727 0.787 48.27

_l-oriented_ 0.417 0.595 43.33 0.709 0.789 45.92
_a-oriented_ 0.419 0.599 43.76 0.717 0.782 45.34
_v-oriented_ 0.418 0.586 42.45 0.713 0.785 46.94

w/o PCCA 0.444 0.554 42.89 0.738 0.775 45.48

Table 3: Ablation study results of MODS’s components on
SIMS and MOSI datasets.

128, 128, 64, 128, the PCCA layers are 3, 3, 3, 4, the co_{_ _}_ _{_ _}_
efficient α are 0.1, 0.1, 0.01, 0.01, and weight decay are
_{_ _}_
1e 3, 1e 3, 1e 2, 1e 2 . The hyper-parameters are
_{_ _−_ _−_ _−_ _−_ _}_
determined based on the validation set.

#### Comparison with State-of-the-art Methods

Tables 1 and 2 present comparison experimental results
of different methods on the MOSI, MOSEI, SIMS, and
SIMSv2 datasets. It can be observed that MODS achieves
the best performance on most metrics across all datasets,
surpassing previous methods to become the new state-ofthe-art (SOTA). Compared to methods that treat all modalities equally or fix a primary one, MODS dynamically selects
the primary modality to reduce interference from heterogeneous emotions. It exhibits lower error rates and stronger
fine-grained emotion discrimination. This observation also
validates the rationality and effectiveness of the sample-level
dynamic primary modality selection approach.

#### Ablation Study

To further verify the effectiveness of the proposed method,
we launched elaborate ablation experiments on both SIMS
and MOSI datasets, as shown in Table 3.
**Effectiveness of GDC Module. For the GDC Module, we**
conduct two comparisons: (1) experiments without GDC,
and (2) experiments excluding the capsule network during graph construction. The results show that two ablation approaches lead to significant performance degradation, with the complete removal of the GDC module causing more severe performance deterioration. This occurs because under unbalanced modality information density, the
selection of primary modalities and cross-modal interactions
are both adversely affected, making it impossible to fully
utilize the effective information from non-language modalities. These findings demonstrate the importance of performing sequence compression on non-language modalities and
highlight the crucial role of using capsule networks for graph
node construction.
**Importance of MSelector Module. To validate the impor-**
tance of the MSelector module, we compare MODS with
models that fixed single modality as the primary modality,
respectively. The experimental results demonstrate that fixing the primary modality leads to performance degradation,


Figure 4: Display of cases and modality weights on SIMS.


indicating that dynamic primary modality selection outperforms static modality assignment.
**Effect of PCCA Module. To validate the proposed PCCA**
module, we compare the model’s performance with and
without PCCA. Specifically, after primary modality selection, we directly perform sentiment regression using only
the selected primary modality, bypassing inter-modal interaction. The consistent performance drop across metrics
demonstrates that even with optimal primary modality selection, ignoring complementary relationships between modalities severely limits prediction accuracy. This confirms that
exhaustive inter-modal interaction, where auxiliary modalities supplement the primary modality with contextual information, is indispensable for achieving robust predictions.

#### Qualitative Evaluation
We illustrate MODS’s dynamic modality selection on SIMS
cases with strong cross-modal conflict. In Figure 4(a), language (“aquarium”) conveys positivity, while audio-visual
cues suggest negativity. MODS emphasizes language, aligning with the weakly positive label. In Figure 4(b), neutral language contrasts with positive acoustic-visual cues.
MODS assigns higher weights to non-language inputs, correctly predicting positive sentiment.

### Conclusion
We propose the MODS algorithm, which provides a more
versatile MSA solution by compressing redundant nonlanguage modality sequences, selecting the primary modality, and progressively enhancing the primary modality. Extensive experiments show the rationality of our work.
**Future Work. The algorithm’s robustness in real-world sce-**
narios will also be considered, such as handling modality data gaps (Yang et al. 2024a) and spurious correlations (Yang et al. 2024c,b).


-----

### Acknowledgments
We sincerely thank Yuxuan Lei and Yue Jiang for their outstanding contributions.

### References

Delbrouck, J.-B.; Tits, N.; Brousmiche, M.; and Dupont,
S. 2020. A transformer-based joint-encoding for emotion recognition and sentiment analysis. _arXiv preprint_
_arXiv:2006.15955._
Drus, Z.; and Khalid, H. 2019. Sentiment analysis in social media and its application: Systematic literature review.
_Procedia Computer Science, 161: 707–714._
Gallagher, C.; Furey, E.; and Curran, K. 2019. The application of sentiment analysis and text analytics to customer
experience reviews to understand what customers are really saying. International Journal of Data Warehousing and
_Mining (IJDWM), 15(4): 21–47._
Han, W.; Chen, H.; Gelbukh, A.; Zadeh, A.; Morency, L.p.; and Poria, S. 2021. Bi-bimodal modality fusion for
correlation-controlled multimodal sentiment analysis. In
_Proceedings of the 2021 international conference on mul-_
_timodal interaction, 6–15._
Han, W.; Chen, H.; and Poria, S. 2021. Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis. arXiv preprint
_arXiv:2109.00412._
Hazarika, D.; Zimmermann, R.; and Poria, S. 2020. Misa:
Modality-invariant and-specific representations for multimodal sentiment analysis. In Proceedings of the 28th ACM
_International Conference on Multimedia, 1122–1131._
Hochreiter, S.; and Schmidhuber, J. 1997. Long short-term
memory. Neural Computation, 9(8): 1735–1780.
Kingma, D. P.; and Ba, J. 2014. Adam: A method for
stochastic optimization. arXiv preprint arXiv:1412.6980.
Kipf, T. N.; and Welling, M. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint
_arXiv:1609.02907._
Lei, Y.; Yang, D.; Li, M.; Wang, S.; Chen, J.; and Zhang,
L. 2023. Text-oriented modality reinforcement network for
multimodal sentiment analysis from unaligned multimodal
sequences. In CAAI International Conference on Artificial
_Intelligence, 189–200. Springer._
Li, Y.; Wang, Y.; and Cui, Z. 2023. Decoupled Multimodal Distilling for Emotion Recognition. In Proceedings
_of the IEEE/CVF Conference on Computer Vision and Pat-_
_tern Recognition, 6631–6640._
Li, Z.; Zhou, Y.; Zhang, W.; Liu, Y.; Yang, C.; Lian, Z.;
and Hu, S. 2022. AMOA: Global acoustic feature enhanced
modal-order-aware network for multimodal sentiment analysis. In Proceedings of the 29th International Conference
_on Computational Linguistics, 7136–7146._
Liang, T.; Lin, G.; Feng, L.; Zhang, Y.; and Lv, F. 2021.
Attention is not enough: Mitigating the distribution discrepancy in asynchronous multimodal sequence fusion. In
_Proceedings of the IEEE/CVF International Conference on_
_Computer Vision, 8148–8156._


Lin, R.; and Hu, H. 2022. Multimodal contrastive learning
via uni-modal coding and cross-modal prediction for multimodal sentiment analysis. arXiv preprint arXiv:2210.14556.
Lin, R.; and Hu, H. 2023. Dynamically shifting multimodal
representations via hybrid-modal attention for multimodal
sentiment analysis. IEEE Transactions on Multimedia, 26:
2740–2755.
Liu, Y.; Yuan, Z.; Mao, H.; Liang, Z.; Yang, W.; Qiu, Y.;
Cheng, T.; Li, X.; Xu, H.; and Gao, K. 2022. Make acoustic
and visual cues matter: Ch-sims v2. 0 dataset and av-mixup
consistent module. In Proceedings of the 2022 international
_conference on multimodal interaction, 247–258._
Liu, Z.; Shen, Y.; Lakshminarasimhan, V. B.; Liang, P. P.;
Zadeh, A.; and Morency, L.-P. 2018. Efficient low-rank multimodal fusion with modality-specific factors. arXiv preprint
_arXiv:1806.00064._
Lv, F.; Chen, X.; Huang, Y.; Duan, L.; and Lin, G. 2021.
Progressive modality reinforcement for human multimodal
emotion recognition from unaligned multimodal sequences.
In Proceedings of the IEEE/CVF Conference on Computer
_Vision and Pattern Recognition, 2554–2562._
Ma, F.; Zhang, Y.; and Sun, X. 2023. Multimodal sentiment analysis with preferential fusion and distance-aware
contrastive learning. In 2023 IEEE International Confer_ence on Multimedia and Expo (ICME), 1367–1372. IEEE._
Mai, S.; Zeng, Y.; Zheng, S.; and Hu, H. 2022. Hybrid contrastive learning of tri-modal representation for multimodal
sentiment analysis. IEEE Transactions on Affective Comput_ing, 14(3): 2276–2289._
Mao, H.; Yuan, Z.; Xu, H.; Yu, W.; Liu, Y.; and Gao, K.
2022. M-SENA: An integrated platform for multimodal sentiment analysis. arXiv preprint arXiv:2203.12441.
Ortis, A.; Farinella, G. M.; and Battiato, S. 2019. An
Overview on Image Sentiment Analysis: Methods, Datasets
and Current Challenges. ICETE (1), 296–306.
Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.;
Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.;
et al. 2019. Pytorch: An imperative style, high-performance
deep learning library. Advances in Neural Information Pro_cessing Systems, 32._
Sabour, S.; Frosst, N.; and Hinton, G. E. 2017. Dynamic
routing between capsules. Advances in neural information
_processing systems, 30._
Sun, Z.; Sarma, P.; Sethares, W.; and Liang, Y. 2020. Learning relationships between text, audio, and video via deep
canonical correlation for multimodal language analysis. In
_Proceedings of the AAAI Conference on Artificial Intelli-_
_gence, volume 34, 8992–8999._
Tsai, M. H.; and Wang, Y. 2021. Analyzing Twitter data
to evaluate people’s attitudes towards public health policies
and events in the era of COVID-19. International Journal of
_Environmental Research and Public Health, 18(12): 6272._
Tsai, Y.-H. H.; Bai, S.; Liang, P. P.; Kolter, J. Z.; Morency,
L.-P.; and Salakhutdinov, R. 2019. Multimodal transformer
for unaligned multimodal language sequences. In Proceed_ings of the conference. Association for Computational Lin-_
_guistics. Meeting, volume 2019, 6558. NIH Public Access._


-----

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in Neural Information Pro_cessing Systems, 30._
Wang, D.; Guo, X.; Tian, Y.; Liu, J.; He, L.; and Luo, X.
2023a. TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis. Pattern Recogni_tion, 136: 109259._
Wang, D.; Liu, S.; Wang, Q.; Tian, Y.; He, L.; and Gao, X.
2022. Cross-modal Enhancement Network for Multimodal
Sentiment Analysis. IEEE Transactions on Multimedia.
Wang, P.; Zhou, Q.; Wu, Y.; Chen, T.; and Hu, J. 2025a.
DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis. In Proceedings of the AAAI Conference on
_Artificial Intelligence, volume 39, 21180–21188._
Wang, Y.; Bi, J.; Ma, Y.; and Pirk, S. 2025b. ASCD:
Attention-Steerable Contrastive Decoding for Reducing
Hallucination in MLLM. arXiv preprint arXiv:2506.14766.
Wang, Y.; Li, Y.; Liang, P. P.; Morency, L.-P.; Bell, P.; and
Lai, C. 2023b. Cross-attention is not enough: Incongruityaware dynamic hierarchical fusion for multimodal affect
recognition. arXiv preprint arXiv:2305.13583.
Wu, S.; Dai, D.; Qin, Z.; Liu, T.; Lin, B.; Cao, Y.; and Sui,
Z. 2023. Denoising Bottleneck with Mutual Information
Maximization for Video Multimodal Fusion. arXiv preprint
_arXiv:2305.14652._
Wu, Y.; Lin, Z.; Zhao, Y.; Qin, B.; and Zhu, L.-N. 2021. A
text-centered shared-private framework via cross-modal prediction for multimodal sentiment analysis. In Findings of the
_Association for Computational Linguistics: ACL-IJCNLP_
_2021, 4730–4738._
Xiao, X.; Yang, J.; and Ning, X. 2021. Research on multimodal emotion analysis algorithm based on deep learning. In Journal of Physics: Conference Series, volume 1802,
032054. IOP Publishing.
Yang, D.; Chen, Z.; Wang, Y.; Wang, S.; Li, M.; Liu, S.;
Zhao, X.; Huang, S.; Dong, Z.; Zhai, P.; and Zhang, L.
2023a. Context De-Confounded Emotion Recognition. In
_Proceedings of the IEEE/CVF Conference on Computer Vi-_
_sion and Pattern Recognition, 19005–19015._
Yang, D.; Huang, S.; Kuang, H.; Du, Y.; and Zhang, L.
2022a. Disentangled Representation Learning for Multimodal Emotion Recognition. In Proceedings of the 30th
_ACM International Conference on Multimedia, 1642–1651._
Yang, D.; Huang, S.; Liu, Y.; and Zhang, L. 2022b. Contextual and Cross-Modal Interaction for Multi-Modal Speech
Emotion Recognition. IEEE Signal Processing Letters, 29:
2093–2097.
Yang, D.; Huang, S.; Wang, S.; Liu, Y.; Zhai, P.; Su, L.; Li,
M.; and Zhang, L. 2022c. Emotion Recognition for Multiple Context Awareness. In Proceedings of the European
_Conference on Computer Vision, volume 13697, 144–162._
Yang, D.; Kuang, H.; Huang, S.; and Zhang, L. 2022d.
Learning Modality-Specific and -Agnostic Representations
for Asynchronous Multimodal Language Sequences. In Pro_ceedings of the 30th ACM International Conference on Mul-_
_timedia, 1708–1717._


Yang, D.; Li, M.; Qu, L.; Yang, K.; Zhai, P.; Wang, S.;
and Zhang, L. 2024a. Asynchronous Multimodal Video
Sequence Fusion via Learning Modality-Exclusive andAgnostic Representations. IEEE Transactions on Circuits
_and Systems for Video Technology._
Yang, D.; Li, M.; Xiao, D.; Liu, Y.; Yang, K.; Chen, Z.;
Wang, Y.; Zhai, P.; Li, K.; and Zhang, L. 2024b. Towards
Multimodal Sentiment Analysis Debiasing via Bias Purification. In Proceedings of the European Conference on Com_puter Vision (ECCV)._
Yang, D.; Liu, Y.; Huang, C.; Li, M.; Zhao, X.; Wang,
Y.; Yang, K.; Wang, Y.; Zhai, P.; and Zhang, L. 2023b.
Target and source modality co-reinforcement for emotion
understanding from asynchronous multimodal sequences.
_Knowledge-Based Systems, 265: 110370._
Yang, D.; Yang, K.; Li, M.; Wang, S.; Wang, S.; and Zhang,
L. 2024c. Robust emotion recognition in context debiasing.
In Proceedings of the IEEE/CVF Conference on Computer
_Vision and Pattern Recognition (CVPR), 12447–12457._
Yu, W.; Xu, H.; Meng, F.; Zhu, Y.; Ma, Y.; Wu, J.; Zou, J.;
and Yang, K. 2020. Ch-sims: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In Proceedings of the 58th Annual Meeting of the Asso_ciation for Computational Linguistics, 3718–3727._
Yu, W.; Xu, H.; Yuan, Z.; and Wu, J. 2021. Learning modality-specific representations with self-supervised
multi-task learning for multimodal sentiment analysis. In
_Proceedings of the AAAI conference on artificial intelli-_
_gence, 10790–10797._
Zadeh, A.; Chen, M.; Poria, S.; Cambria, E.; and Morency,
L.-P. 2017. Tensor fusion network for multimodal sentiment
analysis. arXiv preprint arXiv:1707.07250.
Zadeh, A.; Liang, P. P.; Mazumder, N.; Poria, S.; Cambria,
E.; and Morency, L.-P. 2018a. Memory fusion network for
multi-view sequential learning. In Proceedings of the AAAI
_Conference on Artificial Intelligence, volume 32._
Zadeh, A.; Zellers, R.; Pincus, E.; and Morency, L.-P. 2016.
Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. arXiv preprint
_arXiv:1606.06259._
Zadeh, A. B.; Liang, P. P.; Poria, S.; Cambria, E.; and
Morency, L.-P. 2018b. Multimodal language analysis in the
wild: Cmu-mosei dataset and interpretable dynamic fusion
graph. In Proceedings of the 56th Annual Meeting of the
_Association for Computational Linguistics, 2236–2246._
Zeng, Y.; Mai, S.; Yan, W.; and Hu, H. 2023. Multimodal
reaction: Information modulation for cross-modal representation learning. IEEE Trans. Multimedia.
Zeng, Y.; Yan, W.; Mai, S.; and Hu, H. 2024. Disentanglement Translation Network for multimodal sentiment analysis. Inf. Fusion, 102: 102031.
Zhang, H.; Wang, Y.; Yin, G.; Liu, K.; Liu, Y.; and Yu, T.
2023. Learning language-guided adaptive hyper-modality
representation for multimodal sentiment analysis. _arXiv_
_preprint arXiv:2310.05804._


-----

