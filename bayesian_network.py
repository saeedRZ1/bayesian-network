#!/usr/bin/env python3
"""
bayesian_network.py

A small Bayesian Network implementation and demo (Cloudy / Sprinkler / Rain / WetGrass example).
Features:
- Simple BN structure + CPTs
- Exact inference via enumeration (enumeration_ask)
- CLI demo and example queries

Usage:
    # Run demo
    python3 bayesian_network.py --demo

    # Query example: P(Rain=True | WetGrass=True)
    python3 bayesian_network.py --query "Rain" --evidence "WetGrass=True"

Author: Saeed Razmara
License: MIT
"""
from typing import Dict, List, Tuple, Any
import argparse

# Node names
# Cloudy -> Sprinkler, Rain -> WetGrass depends on Sprinkler and Rain
NODES = ["Cloudy", "Sprinkler", "Rain", "WetGrass"]

# Conditional probability tables (CPTs)
# Represented as functions that take a dict of parent values and return P(node=True)
def p_cloudy(parents: Dict[str, bool]) -> float:
    # P(Cloudy = True) = 0.5
    return 0.5

def p_sprinkler(parents: Dict[str, bool]) -> float:
    # parents: {"Cloudy": bool}
    return 0.1 if parents.get("Cloudy", False) else 0.5

def p_rain(parents: Dict[str, bool]) -> float:
    # parents: {"Cloudy": bool}
    return 0.8 if parents.get("Cloudy", False) else 0.2

def p_wetgrass(parents: Dict[str, bool]) -> float:
    # parents: {"Sprinkler": bool, "Rain": bool}
    s = parents.get("Sprinkler", False)
    r = parents.get("Rain", False)
    if s and r:
        return 0.99
    elif s and not r:
        return 0.9
    elif not s and r:
        return 0.9
    else:
        return 0.0

# Map node -> (parents, CPT function)
BAYES_NET = {
    "Cloudy": ([], p_cloudy),
    "Sprinkler": (["Cloudy"], p_sprinkler),
    "Rain": (["Cloudy"], p_rain),
    "WetGrass": (["Sprinkler", "Rain"], p_wetgrass),
}

def prob_of(node: str, value: bool, assignment: Dict[str, bool]) -> float:
    """Return P(node=value | parents in assignment)."""
    parents, cpt = BAYES_NET[node]
    parent_vals = {p: assignment[p] for p in parents}
    p_true = cpt(parent_vals)
    return p_true if value else 1 - p_true

def joint_prob(assignment: Dict[str, bool]) -> float:
    """Compute joint probability of a full assignment over all nodes."""
    p = 1.0
    for node in NODES:
        val = assignment[node]
        p *= prob_of(node, val, assignment)
    return p

def enumerate_all(variables: List[str], evidence: Dict[str, bool]) -> float:
    """Recursive enumeration to compute probability of evidence (marginalizing others)."""
    if not variables:
        return 1.0
    first, rest = variables[0], variables[1:]
    if first in evidence:
        return prob_of(first, evidence[first], evidence) * enumerate_all(rest, evidence)
    else:
        total = 0.0
        for val in [True, False]:
            new_evidence = dict(evidence)
            new_evidence[first] = val
            total += prob_of(first, val, new_evidence) * enumerate_all(rest, new_evidence)
        return total

def enumeration_ask(query: str, evidence: Dict[str, bool]) -> Dict[bool, float]:
    """
    Return distribution P(query | evidence) as dict {True: p, False: p}
    using enumeration.
    """
    # Normalize
    Q = {}
    for val in [True, False]:
        extended = dict(evidence)
        extended[query] = val
        Q[val] = enumerate_all(NODES, extended)
    total = Q[True] + Q[False]
    if total == 0:
        return {True: 0.0, False: 0.0}
    return {True: Q[True] / total, False: Q[False] / total}

def parse_assignment(s: str) -> Dict[str, bool]:
    """
    Parse evidence string like "WetGrass=True,Cloudy=False" into dict.
    """
    assignment = {}
    if not s:
        return assignment
    parts = s.split(",")
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=")
        k = k.strip()
        v = v.strip().lower()
        if k not in NODES:
            raise ValueError(f"Unknown variable: {k}")
        if v in ("true", "1", "t"):
            assignment[k] = True
        elif v in ("false", "0", "f"):
            assignment[k] = False
        else:
            raise ValueError(f"Unknown boolean value: {v}")
    return assignment

def demo():
    print("Bayesian Network demo (Cloudy, Sprinkler, Rain, WetGrass)")
    print("Network structure:")
    for node, (parents, _) in BAYES_NET.items():
        print(f"  {node} parents: {parents}")
    print("\nExample queries:")
    examples = [
        ("Rain", {"WetGrass": True}),
        ("Cloudy", {"WetGrass": True}),
        ("Sprinkler", {"WetGrass": True}),
        ("Rain", {"Sprinkler": True}),
    ]
    for q, ev in examples:
        dist = enumeration_ask(q, ev)
        print(f"P({q}=True | {ev}) = {dist[True]:.4f}, P(False)={dist[False]:.4f}")
    print("\nExact joint probability example (Cloudy=True, Sprinkler=False, Rain=True, WetGrass=True):")
    assign = {"Cloudy": True, "Sprinkler": False, "Rain": True, "WetGrass": True}
    print(f"Joint: {joint_prob(assign):.8f}")

def main():
    parser = argparse.ArgumentParser(description="Simple Bayesian Network demo & queries")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--query", type=str, help='Query variable name, e.g. "Rain"')
    parser.add_argument("--evidence", type=str, default="", help='Evidence like "WetGrass=True,Cloudy=False"')
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if args.query:
        evidence = parse_assignment(args.evidence)
        result = enumeration_ask(args.query, evidence)
        print(f"P({args.query}=True | {evidence}) = {result[True]:.6f}")
        print(f"P({args.query}=False | {evidence}) = {result[False]:.6f}")
    else:
        demo()

if __name__ == "__main__":
    main()
