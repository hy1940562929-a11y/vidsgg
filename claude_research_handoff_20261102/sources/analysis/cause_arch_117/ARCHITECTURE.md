# E117 architecture note
Intervention is **mask-loss weighting only**. Matching, class CE, hull decode unchanged.
`e117_centerness.centerness_loss_mask` builds a Gaussian on GT span; `losses_arm` swaps `loss_mask` for focal+dice.
