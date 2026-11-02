# E116-SPAN-AUX-HULL
N native; T +span L1+GIoU aux on matched queries; X same aux with shuffled GT cxw.
Eval always mask>0.5 hull (NOT span decode). eta>=0.20; all seed T-N>0 and T-X>0.
Admission: rebuild; C0 46/29; zero-init span inactive on masks. VERIFIED or REJECT.
Deconfounds H113 (aux without decode swap).
