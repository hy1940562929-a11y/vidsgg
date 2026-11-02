# E114 architecture note

Intervention is **matching-only**. Forward, loss_labels, loss_masks, and hull decode unchanged.
`e114_match.bipartite_match_tgiou` mirrors MaskVRD.bipartite_match then adds DETR-style temporal GIoU cost.
