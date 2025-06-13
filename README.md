# Mini Project 2: Spatial Influence on Dynamics of False Information Spread. 
Extension of Mini Project 1. People are nodes and a link is a spatial/virtual connection between 2 people. 
Information can only spread on links. 

**Dynamic Model Equations:** <br>
$p_{b, t} = p_{b, t-1} + (p_{nb} \cdot Max[(p_{b, t-1}-p_{n, t-1}), 0] \cdot p_{n, t-1}) - (p_{bn} \cdot Max[(p_{n, t-1}-p_{b, t-1}), 0] \cdot p_{b, t-1})$

$p_{n, t} = p_{n, t-1} + (p_{bn} \cdot Max[(p_{n, t-1}-p_{b, t-1}), 0] \cdot p_{b, t-1}) - (p_{nb} \cdot Max[(p_{b, t-1}-p_{n, t-1}), 0] \cdot p_{n, t-1})$ 

$p_{neutral, t} = 1 - p_{b, t} - p_{n, t}$

## Description

Right now, outputs a single random undirected graph using NetworkX at different time steps w/ proportions of believers, non-believers, and neutrals displayed over time based on user inputted # of nodes. <br>
Next steps?? <br>
- Fix initial graph to be connected.
- changing initial proportion values
- dropdowns for changing seed which essentially changes network topology - seed is for reproducibility of random experiments
- changing rate

## References
We acknowledge the use of AI in generating code. 
