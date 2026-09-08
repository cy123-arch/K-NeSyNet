import argparse, json, torch
from knesynet.kg.build import build_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", default="data/graph.pt")
    args = ap.parse_args()
    g = build_graph(args.entities, args.edges)
    torch.save({
        "edge_index": g.edge_index,
        "edge_type": g.edge_type,
        "entity_to_idx": g.entity_to_idx,
        "relation_to_idx": g.relation_to_idx,
    }, args.out)
    print(f"saved {len(g.idx_to_entity)} entities, {g.edge_index.shape[1]} edges to {args.out}")

if __name__ == "__main__":
    main()
