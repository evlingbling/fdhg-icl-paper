from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def add_column_vertices(G: nx.MultiDiGraph, table_name: str, columns: list[str]) -> None:
    for col in columns:
        node = f"{table_name}.{col}"
        G.add_node(node, table=table_name, column=col, node_type="column")


def add_schema_fd_edges(G: nx.MultiDiGraph, table_name: str, key_col: str, columns: list[str]) -> None:
    lhs = f"{table_name}.{key_col}"
    for col in columns:
        if col == key_col:
            continue
        rhs = f"{table_name}.{col}"
        G.add_edge(
            lhs,
            rhs,
            edge_type="schema_fd",
            weight=1.0,
            label="schema_fd",
            stats={"declared": True},
        )


def add_fk_edges(
    G: nx.MultiDiGraph,
    child_table: str,
    child_fk: str,
    parent_table: str,
    parent_pk: str,
) -> None:
    child_node = f"{child_table}.{child_fk}"
    parent_node = f"{parent_table}.{parent_pk}"

    # FK/IND: child FK points to parent PK.
    G.add_edge(
        child_node,
        parent_node,
        edge_type="fk_ind",
        weight=1.0,
        label="FK/IND",
        stats={"declared": True},
    )

    # Inverse-FK: parent entity can aggregate child rows.
    G.add_edge(
        parent_node,
        child_node,
        edge_type="inverse_fk",
        weight=1.0,
        label="inverse_FK_agg",
        stats={"aggregation_template": True},
    )


def add_afd_edges_from_csv(G: nx.MultiDiGraph, table_name: str, afd_edges_path: Path) -> None:
    edges = pd.read_csv(afd_edges_path)

    for _, row in edges.iterrows():
        lhs = f"{table_name}.{row['lhs']}"
        rhs = f"{table_name}.{row['rhs']}"
        G.add_edge(
            lhs,
            rhs,
            edge_type="afd",
            weight=float(row.get("edge_weight", 0.0)),
            label=f"AFD {row['lhs']}→{row['rhs']}",
            stats={
                "r_del": float(row.get("r_del", 0.0)),
                "r_ent": float(row.get("r_ent", 0.0)),
                "lhs_uniqueness": float(row.get("lhs_uniqueness", 0.0)),
                "lhs_support_repeated": float(row.get("lhs_support_repeated", 0.0)),
                "coverage": float(row.get("coverage", 0.0)),
            },
        )


def graph_to_json(G: nx.MultiDiGraph) -> dict:
    nodes = [
        {"id": node, **attrs}
        for node, attrs in G.nodes(data=True)
    ]

    edges = []
    for u, v, key, attrs in G.edges(keys=True, data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "key": key,
                **attrs,
            }
        )

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    }


def save_visualization(G: nx.MultiDiGraph, out_png: Path, title: str) -> None:
    plt.figure(figsize=(14, 9))

    pos = nx.spring_layout(G, seed=41, k=1.2)

    edge_types = nx.get_edge_attributes(G, "edge_type")
    node_tables = nx.get_node_attributes(G, "table")

    color_map = {
        "product": "#A7C7E7",
        "review": "#B7E4C7",
        "customer": "#FFD6A5",
        "users": "#A7C7E7",
        "products": "#B7E4C7",
        "events": "#FFD6A5",
    }

    node_colors = [
        color_map.get(node_tables.get(n, ""), "#DDDDDD")
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1800,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n.replace(".", "\n") for n in G.nodes()},
        font_size=8,
    )

    # Draw edges by type.
    style_map = {
        "schema_fd": "solid",
        "fk_ind": "solid",
        "inverse_fk": "dashed",
        "afd": "dotted",
    }

    for edge_type in sorted(set(edge_types.values())):
        edgelist = [
            (u, v)
            for u, v, k in G.edges(keys=True)
            if G.edges[u, v, k].get("edge_type") == edge_type
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            width=1.8 if edge_type in {"fk_ind", "inverse_fk"} else 1.2,
            style=style_map.get(edge_type, "solid"),
            connectionstyle="arc3,rad=0.10",
        )

    edge_labels = {}
    for u, v, k, attrs in G.edges(keys=True, data=True):
        label = attrs.get("label", attrs.get("edge_type", ""))
        if attrs.get("edge_type") == "afd":
            label += f"\nw={attrs.get('weight', 0):.2f}"
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_real_graph(args: argparse.Namespace) -> nx.MultiDiGraph:
    product = pd.read_parquet(args.product_table)
    review = pd.read_parquet(args.review_table)

    G = nx.MultiDiGraph(name="real_rel_amazon_item_churn_fdhg")

    add_column_vertices(G, "product", list(product.columns))
    add_column_vertices(G, "review", list(review.columns))

    add_schema_fd_edges(G, "product", "product_id", list(product.columns))
    add_schema_fd_edges(G, "review", "product_id", ["product_id", "rating", "review_time"])

    add_fk_edges(
        G,
        child_table="review",
        child_fk="product_id",
        parent_table="product",
        parent_pk="product_id",
    )

    add_afd_edges_from_csv(
        G,
        table_name="product",
        afd_edges_path=Path(args.afd_edges),
    )

    return G


def build_synthetic_graph(args: argparse.Namespace) -> nx.MultiDiGraph:
    users = pd.read_parquet(Path(args.synthetic_dir) / "table_users.parquet")
    products = pd.read_parquet(Path(args.synthetic_dir) / "table_products.parquet")
    events = pd.read_parquet(Path(args.synthetic_dir) / "table_events.parquet")

    G = nx.MultiDiGraph(name="synthetic_minimal_fdhg")

    add_column_vertices(G, "users", list(users.columns))
    add_column_vertices(G, "products", list(products.columns))
    add_column_vertices(G, "events", list(events.columns))

    add_schema_fd_edges(G, "users", "user_id", list(users.columns))
    add_schema_fd_edges(G, "products", "product_id", list(products.columns))
    add_schema_fd_edges(G, "events", "event_id", list(events.columns))

    add_fk_edges(G, "events", "user_id", "users", "user_id")
    add_fk_edges(G, "events", "product_id", "products", "product_id")

    # Ground-truth AFDs from synthetic prior.
    G.add_edge(
        "users.zip",
        "users.city",
        edge_type="afd",
        weight=0.99,
        label="AFD zip→city",
        stats={"ground_truth": True},
    )

    G.add_edge(
        "products.brand",
        "products.category",
        edge_type="afd",
        weight=0.95,
        label="AFD brand→category",
        stats={"ground_truth": True},
    )

    return G


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "synthetic"], required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--product-table", default="outputs/relbench_inspect/rel-amazon_item-churn/table_product.parquet")
    parser.add_argument("--review-table", default="outputs/relbench_inspect/rel-amazon_item-churn/table_review.parquet")
    parser.add_argument("--afd-edges", default="outputs/ambiguity_features/rel-amazon_item-churn_sample/afd_edges_product_dmax1.csv")

    parser.add_argument("--synthetic-dir", default="outputs/synthetic_prior/minimal_seed41")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "real":
        G = build_real_graph(args)
        stem = "real_rel_amazon_item_churn_fdhg"
        title = "FDHG graph: rel-amazon/item-churn MVP"
    else:
        G = build_synthetic_graph(args)
        stem = "synthetic_minimal_fdhg"
        title = "FDHG graph: minimal synthetic prior"

    graph_json = graph_to_json(G)

    json_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"

    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=2)

    save_visualization(G, png_path, title)

    print("=== FDHG graph built ===")
    print("mode:", args.mode)
    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    print("saved json:", json_path)
    print("saved png:", png_path)

    print("\nEdge summary:")
    for _, _, _, attrs in G.edges(keys=True, data=True):
        print("-", attrs.get("edge_type"), attrs.get("label"), "weight=", attrs.get("weight"))


if __name__ == "__main__":
    main()
