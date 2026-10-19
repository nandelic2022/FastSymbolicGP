from __future__ import annotations
from .program import Node, Program, simplify_node


def _same(a:Node,b:Node) -> bool:
    return a.to_string()==b.to_string()


def _flatten(name,node):
    if node.kind=="func" and node.value==name and len(node.children)==2:
        return _flatten(name,node.children[0])+_flatten(name,node.children[1])
    return [node]


def _balanced(name,nodes):
    nodes=list(nodes)
    if not nodes: return Node("const",0.0 if name=="add" else 1.0)
    while len(nodes)>1:
        nxt=[]
        for i in range(0,len(nodes),2):
            nxt.append(nodes[i] if i+1>=len(nodes) else Node("func",name,(nodes[i],nodes[i+1])))
        nodes=nxt
    return nodes[0]


def canonicalize_node(node:Node, mode="canonical") -> Node:
    if node.kind!="func": return node.clone()
    cs=tuple(canonicalize_node(c,mode) for c in node.children); name=str(node.value)
    n=simplify_node(Node("func",name,cs,node.unit))
    if n.kind!="func": return n
    cs=n.children; name=str(n.value)
    # Safe idempotent rewrites.
    if name=="neg" and cs[0].kind=="func" and cs[0].value=="neg": return cs[0].children[0].clone()
    if name=="abs" and cs[0].kind=="func" and cs[0].value=="abs": return cs[0].clone()
    if name in {"min","max"} and _same(cs[0],cs[1]): return cs[0].clone()
    if name=="sub" and cs[1].kind=="const" and abs(float(cs[1].value))<1e-15: return cs[0].clone()
    if name=="div" and cs[1].kind=="const" and abs(float(cs[1].value)-1.0)<1e-15: return cs[0].clone()
    # Associative/commutative canonical form makes duplicate detection stronger.
    if name in {"add","mul"}:
        terms=[]
        for c in cs: terms.extend(_flatten(name,c))
        terms=sorted((canonicalize_node(t,mode) for t in terms),key=lambda x:x.to_string())
        return _balanced(name,terms)
    if name in {"min","max"}:
        cs=tuple(sorted(cs,key=lambda x:x.to_string()))
        return Node("func",name,cs,n.unit)
    if str(mode).lower() in {"aggressive","algebraic"}:
        if name=="sub" and _same(cs[0],cs[1]): return Node("const",0.0)
        if name=="div" and _same(cs[0],cs[1]): return Node("const",1.0)
        if name=="add" and _same(cs[0],cs[1]): return Node("func","mul",(Node("const",2.0),cs[0].clone()))
    return Node("func",name,cs,n.unit)


def canonicalize_program(program:Program, mode="canonical", max_passes: int=6) -> Program:
    q=program.clone()
    previous=None
    for _ in range(max(1,int(max_passes))):
        q.root=canonicalize_node(q.root,mode)
        text=q.root.to_string()
        if text==previous: break
        previous=text
    q.invalidate_caches()
    q.metadata["canonicalized"]=True
    q.metadata["canonicalization_mode"]=str(mode)
    return q
