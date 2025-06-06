import torch
from numpy import array, dot
from numpy.linalg import norm
def cosine_similarity(a, b):
    a, b = array(a), array(b)
    return dot(a, b) / (norm(a) * norm(b) + 1e-8)
def heatmap_coverage(grad_vector, forget_nodes, top_ratio: float = 0.10):
    grad = array(grad_vector)
    k = max(1, int(len(grad) * top_ratio))
    top_idx = set(grad.argsort()[-k:])
    forget = {int(n.item()) if isinstance(n, torch.Tensor) else int(n) for n in forget_nodes}
    if len(forget) == 0:
        return 0.0
    covered = sum(1 for i in forget if i in top_idx)
    return round(covered / len(forget), 4)
def normalize_rule(rule):
    return rule.strip().replace(" ", "").lower()
def evaluate_metrics(
    chef_pre,
    chef_post,
    proxy_pre,
    proxy_post,
    ra_pre,
    ra_post,
    grad_pre,
    grad_post,
    model,
    data,
    unlearn_time,
    forget_nodes,
):
    rules_pre = set(normalize_rule(r) for r in chef_pre["rules"])
    rules_post = set(normalize_rule(r) for r in chef_post["rules"])
    rules_removed = len(rules_pre - rules_post) 
    jaccard_similarity = len(rules_pre.intersection(rules_post)) / len(rules_pre.union(rules_post))
    edges_pre = set(tuple(sorted(e)) for e in proxy_pre["graph"])
    edges_post = set(tuple(sorted(e)) for e in proxy_post["graph"])
    try:
        import networkx as nx

        G_pre = nx.Graph()
        G_pre.add_edges_from(edges_pre)
        G_post = nx.Graph()
        G_post.add_edges_from(edges_post)
        ged_iter = nx.algorithms.similarity.graph_edit_distance(
            G_pre, G_post, timeout=5
        )
        ged_val = next(ged_iter) 
        if ged_val is None: 
            raise ValueError("GED timeout")
    except Exception as e:
        print(f"Warning: graph_edit_distance fallback ({e})")
        ged_val = len(edges_pre.symmetric_difference(edges_post))
    grad_pre_arr = array(grad_pre)
    grad_post_arr = array(grad_post)
    min_len = min(len(grad_pre_arr), len(grad_post_arr))
    grad_pre_arr = grad_pre_arr[:min_len]
    grad_post_arr = grad_post_arr[:min_len]
    hs = round(1 - cosine_similarity(grad_pre_arr, grad_post_arr), 4)
    esd = round(norm(grad_pre_arr - grad_post_arr), 4)
    hc_pre = heatmap_coverage(grad_pre, forget_nodes)
    hc_post = heatmap_coverage(grad_post, forget_nodes)
    return {
        "RA (Pre)": f"{ra_pre}%",
        "RA (Post)": f"{ra_post}%",
        "RA Δ": f"{round(ra_pre - ra_post, 2)}%",
        "HS": hs,
        "HC (Pre)": hc_pre,
        "HC (Post)": hc_post,
        "HC Δ": round(hc_pre - hc_post, 4),
        "ESD": esd,
        "GED (Pre)": len(edges_pre),
        "GED (Post)": len(edges_post),
        "GED Δ": ged_val,
        "Rules (Pre)": len(rules_pre),
        "Rules (Post)": len(rules_post),
        "Rules Removed": rules_removed,
        "Unlearn Time (s)": round(unlearn_time, 2),
        "Model": type(model).__name__,
        "Dataset": "physics",
    }
def calculate_residual_attribution(model, data, nodes):
    device = data.x.device
    data.x = data.x.detach().clone()
    data.x.requires_grad = True
    model.eval()
    out = model(data.x, data.edge_index)
    loss = out.mean()
    loss.backward()
    grad = data.x.grad.abs().sum(dim=1).detach().cpu() 
    total = grad.sum().item()
    nodes = nodes.cpu()
    forget_score = grad[nodes].sum().item()
    ra = round((forget_score / total) * 100, 2)
    return ra, grad.tolist()
