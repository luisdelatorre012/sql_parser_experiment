"""
sql_subquery_cte_transformer
============================

Bidirectional transformation between inline sub-queries and
Common-Table-Expressions (CTEs), using only sqlglot’s AST.

Public API
----------
transform_subqueries_to_ctes(sql: str, *, dialect: str | None = None) -> str
transform_ctes_to_subqueries(sql: str, *, dialect: str | None = None) -> str
MultipleQueriesError
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set

import sqlglot
from sqlglot import exp
from sqlglot.dialects import DIALECTS
from sqlglot.expressions import alias_

DEFAULT_DIALECT: str | None = None  # e.g. "postgres", "mysql", …


class MultipleQueriesError(Exception):
    """Raised when more than one top-level SQL statement is supplied."""


def _validate_dialect(name: str | None) -> str | None:
    if name is None:
        return DEFAULT_DIALECT
    n = name.lower()
    if n not in {d.lower() for d in DIALECTS}:
        raise ValueError(
            f"Unknown sqlglot dialect '{name}'. "
            f"Choose one of: {', '.join(sorted(DIALECTS))}"
        )
    return n


@dataclass
class _TransformState:
    counter: int = 0
    mapping: Dict[str, str] = field(default_factory=dict)   # normalized_sql → cte_n
    cte_definitions: List[str] = field(default_factory=list)
    scalar_aliases: Set[str] = field(default_factory=set)   # names of 1×1 CTEs
    used_names:   Set[str] = field(default_factory=set)     # top-level tables & aliases

    def next_alias(self) -> str:
        while True:
            name = f"cte_{self.counter}"
            self.counter += 1
            if name.upper() not in self.used_names:
                self.used_names.add(name.upper())
                return name


def _normalize_sql(sql: str) -> str:
    # collapse whitespace + uppercase for stable dedupe keys
    return " ".join(sql.split()).upper()


def _tables_in_from(frm: exp.From) -> set[str]:
    names: set[str] = set()
    if isinstance(frm.this, exp.Table):
        names.add(frm.this.name.upper())
    for j in frm.args.get("joins", []):
        if isinstance(j.this, exp.Table):
            names.add(j.this.name.upper())
    return names


def _outer_table_aliases(sql: str, *, dialect: str | None) -> set[str]:
    """
    Return just the top-level tables & aliases in the outer query's FROM/JOIN.
    """
    root = sqlglot.parse_one(sql, read=dialect)
    names: set[str] = set()
    frm = root.args.get("from")
    if isinstance(frm, exp.From):
        tbl = frm.this
        if isinstance(tbl, exp.Table):
            names.add(tbl.name.upper())
            if tbl.alias and tbl.alias.name:
                names.add(tbl.alias.name.upper())
        for j in frm.args.get("joins", []):
            t = j.this
            if isinstance(t, exp.Table):
                names.add(t.name.upper())
                if t.alias and t.alias.name:
                    names.add(t.alias.name.upper())
    return names


def _is_scalar_query(sel: exp.Select) -> bool:
    # aggregate w/o GROUP BY or single literal/anonymous
    if sel.args.get("group") is None and any(isinstance(n, exp.AggFunc) for n in sel.find_all(exp.AggFunc)):
        return True
    exps = sel.expressions
    return len(exps) == 1 and isinstance(exps[0], (exp.Literal, exp.Anonymous))


def _contains_set_op(node: exp.Expression) -> bool:
    return bool(node.find(exp.Union) or node.find(exp.Intersect) or node.find(exp.Except))


def _contains_exists(node: exp.Expression) -> bool:
    return node.find(exp.Exists) is not None


def _contains_window(node: exp.Expression) -> bool:
    return node.find(exp.Window) is not None


def _make_cte_sql(inner_sql: str, alias: str, *, dialect: str | None) -> str:
    sel = sqlglot.parse_one(inner_sql, read=dialect)
    if isinstance(sel, exp.Select) and _is_scalar_query(sel):
        proj = sel.expressions[0]
        if not proj.alias_or_name:
            sel.set("expressions", [alias_(proj.copy(), "val")])
    body = sel.sql(dialect=dialect).rstrip(";")
    return f"{alias} AS ({body})"


def transform_subqueries_to_ctes(
    sql: str,
    *,
    dialect: str | None = None,
) -> str:
    dialect = _validate_dialect(dialect)

    # 1) Ensure exactly one statement
    stmts = sqlglot.parse(sql, read=dialect)
    if len(stmts) != 1:
        raise MultipleQueriesError("Exactly one SQL statement is required.")
    root = stmts[0]

    # 2) Prepare state & record outer table names
    raw = sql
    state = _TransformState()
    state.used_names.update(_outer_table_aliases(raw, dialect=dialect))

    # 3) Hoist derived tables (FROM (SELECT…) alias)
    for parent in root.find_all((exp.From, exp.Join)):
        sub = parent.this
        if not isinstance(sub, exp.Subquery) or not isinstance(sub.this, exp.Select):
            continue
        alias_exp = parent.args.get("alias")
        if not alias_exp or not alias_exp.name:
            continue
        inner = sub.this.sql(dialect=dialect)
        key = _normalize_sql(inner)
        if key not in state.mapping:
            cte = state.next_alias()
            state.mapping[key] = cte
            state.cte_definitions.append(_make_cte_sql(inner, cte, dialect=dialect))
        parent.set("this", exp.Table(this=state.mapping[key]))

    # 4) Hoist scalar sub-queries in projections / predicates
    def _lift_scalar(node: exp.Expression):
        if not isinstance(node, exp.Subquery) or not isinstance(node.this, exp.Select):
            return node
        # skip inline-view contexts
        if isinstance(node.parent, (exp.From, exp.Join)):
            return node

        sel = node.this
        scalar = _is_scalar_query(sel)
        # skip disallowed subqueries
        if _contains_set_op(sel) or _contains_exists(sel) or (scalar and _contains_window(sel)):
            return node
        # skip correlated scalars
        if scalar:
            refs = {t.name.upper() for t in sel.find_all(exp.Table)}
            if refs & state.used_names:
                return node

        inner = sel.sql(dialect=dialect)
        key = _normalize_sql(inner)
        if key not in state.mapping:
            cte = state.next_alias()
            state.mapping[key] = cte
            state.cte_definitions.append(_make_cte_sql(inner, cte, dialect=dialect))
            if scalar:
                state.scalar_aliases.add(cte)

        cte_name = state.mapping[key]
        # always return cte_n.val; in SELECT this becomes `cte_n.val AS alias`
        return exp.column("val", table=cte_name)

    root = root.transform(_lift_scalar)

    # 5) Inject CROSS JOINs for all scalar CTEs
    if state.scalar_aliases:
        frm = root.args.get("from")
        if not isinstance(frm, exp.From):
            first, *rest = state.scalar_aliases
            root.set("from", exp.From(this=exp.Table(this=first)))
            pending = set(rest)
        else:
            existing = _tables_in_from(frm)
            pending = {a for a in state.scalar_aliases if a.upper() not in existing}

        for alias in pending:
            root.append("joins", exp.Join(this=exp.Table(this=alias), kind="cross"))

    # 6) Pretty-print & prepend WITH
    out = root.sql(dialect=dialect, pretty=True)
    if state.cte_definitions:
        with_clause = "WITH " + ",\n     ".join(state.cte_definitions)
        out = f"{with_clause}\n{out}"

    return out


def transform_ctes_to_subqueries(
    sql: str,
    *,
    dialect: str | None = None,
) -> str:
    dialect = _validate_dialect(dialect)
    root = sqlglot.parse_one(sql, read=dialect)
    with_clause = root.args.get("with")
    if not isinstance(with_clause, exp.With):
        return sqlglot.parse_one(sql, read=dialect).sql(pretty=True)

    # build alias→AST map
    cte_map: Dict[str, exp.Select] = {}
    scalar_aliases: Set[str] = set()
    for cte in with_clause.find_all(exp.CTE):
        a = cte.alias_or_name
        cte_map[a] = cte.this
        if _is_scalar_query(cte.this):
            scalar_aliases.add(a)

    root.set("with", None)

    def _inline(node: exp.Expression):
        # drop CROSS JOIN cte_n
        if (isinstance(node, exp.Join)
                and node.kind.upper() == "CROSS"
                and isinstance(node.this, exp.Table)
                and node.this.name in cte_map):
            return False
        # inline table
        if isinstance(node, exp.Table) and node.name in cte_map:
            return exp.Subquery(this=cte_map[node.name].copy()).alias(node.alias_or_name)
        # inline scalar column
        if isinstance(node, exp.Column) and node.table in scalar_aliases:
            return exp.Subquery(this=cte_map[node.table].copy())
        return node

    root = root.transform(_inline)
    return root.sql(pretty=True)
