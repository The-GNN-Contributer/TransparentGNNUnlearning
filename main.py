import pandas as pd
from explainers import run_graphchef, run_proxy_graph_generator
from unlearning import prepare_pre_unlearning, perform_unlearning
from metrics import calculate_residual_attribution, evaluate_metrics
from visualization import (
    export_results,
    plot_rule_diff,
    plot_graph_diff,
    plot_ra_per_node,
    plot_explanation_shift,
    plot_rule_counts,
    visualize_graphchef_rules,
    compare_graphchef_trees,
    plot_attribution_histogram,      
    plot_attribution_scatter,
    plot_batch_comparison           
)
def normalize_rule(rule):
    return rule.strip().replace(" ", "").lower()
def get_removed_and_modified_rules(chef_pre_rules, chef_post_rules):
    removed_rules = []
    modified_rules = []
    pre_rule_set = set(chef_pre_rules)
    post_rule_set = set(chef_post_rules)
    removed_rules = pre_rule_set - post_rule_set
    for rule_pre in chef_pre_rules:
        if rule_pre not in chef_post_rules:
            for rule_post in chef_post_rules:
                if rule_pre.split('→')[1] == rule_post.split('→')[1]: 
                    modified_rules.append(rule_pre) 
    return removed_rules, modified_rules
def run_pipeline():
    model, data, forget_nodes = prepare_pre_unlearning()
    ra_pre, grad_pre = calculate_residual_attribution(model, data, forget_nodes)
    chef_pre = run_graphchef(stage="pre", model=model, data=data)
    proxy_pre = run_proxy_graph_generator(
        stage="pre", model=model, data=data, target_nodes=forget_nodes
    )
    model, data, unlearn_time, mapped_forget_nodes = perform_unlearning(
        strategy="idea", model=model, data=data, forget_nodes=forget_nodes
    )
    chef_post = run_graphchef(stage="post", model=model, data=data)
    proxy_post = run_proxy_graph_generator(
        stage="post", model=model, data=data, target_nodes=forget_nodes
    )
    ra_post, grad_post = calculate_residual_attribution(model, data, mapped_forget_nodes)
    results = evaluate_metrics(
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
    )
    removed_rules, modified_rules = get_removed_and_modified_rules(chef_pre["rules"], chef_post["rules"])
    print(f"Rules before unlearning:")
    normalized_rules_pre = [normalize_rule(r) for r in chef_pre["rules"]]
    print("\n".join(normalized_rules_pre))
    print(f"\nRules after unlearning:")
    normalized_rules_post = [normalize_rule(r) for r in chef_post["rules"]]
    print("\n".join(normalized_rules_post))
    print(f"\nRemoved Rules: {len(removed_rules)}")
    print("\n".join(removed_rules))
    print(f"\nModified Rules: {len(modified_rules)}")
    print("\n".join(modified_rules))
    #plot_ra_per_node(forget_nodes, grad_pre, grad_post)
    #plot_explanation_shift(grad_pre, grad_post)
    #plot_rule_diff(chef_pre["rules"], chef_post["rules"])
    #plot_graph_diff(proxy_pre["graph"], proxy_post["graph"])
    #plot_rule_counts(chef_pre["rules"], chef_post["rules"])
    #plot_attribution_histogram(grad_pre, grad_post)
    #plot_attribution_scatter(grad_pre, grad_post)
    export_results(results)
    feature_names = [f"feature_{i}" for i in range(data.x.shape[1])]
    #visualize_graphchef_rules(chef_pre["clf"], feature_names)
    #compare_graphchef_trees(chef_pre["clf"], chef_post["clf"], feature_names)

if __name__ == "__main__":
    run_pipeline()
