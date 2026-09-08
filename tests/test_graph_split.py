import pandas as pd
from knesynet.kg.splits import remove_heldout_treatment_edges

def test_remove_heldout_treatment_edges():
    e = pd.DataFrame([
        {"src":"P1","relation":"treated_with","dst":"D1"},
        {"src":"P2","relation":"treated_with","dst":"D2"},
        {"src":"P2","relation":"has_mutation","dst":"G1"},
    ])
    kept, removed = remove_heldout_treatment_edges(e, ["P2"])
    assert len(removed) == 1
    assert ((kept["src"]=="P2") & (kept["relation"]=="has_mutation")).any()
