"""
sql_subquery_cte_transformer
============================

Bidirectional transformation between inline sub‑queries and
Common Table Expressions (CTEs).

Public API
----------
transform_subqueries_to_ctes(sql: str, *, dialect: str | None = None) -> str
transform_ctes_to_subqueries(sql: str, *, dialect: str | None = None) -> str
MultipleQueriesError
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

import sqlglot
import sqlparse
from sqlparse import tokens as sqlparse_tokens
from sqlglot.expressions import alias_
from sqlglot.dialects import DIALECTS
from sqlglot import exp

DEFAULT_DIALECT: str | None = None  # change to "postgres", "mysql", … if desired


# ---------------------------------------------------------------------------#
# Public exceptions
# ---------------------------------------------------------------------------#


class MultipleQueriesError(Exception):
    """Raised when more than one top‑level SQL statement is supplied."""


# ---------------------------------------------------------------------------#
# Regexes – pre‑compiled for speed
# ---------------------------------------------------------------------------#

RE_SET_OP = re.compile(r"\b(?:UNION|INTERSECT|EXCEPT)\b", re.I)
RE_EXISTS = re.compile(r"\b(?:NOT\s+)?EXISTS\b", re.I)
RE_WINDOW = re.compile(r"\bOVER\s*\(", re.I)
RE_FROM_OR_JOIN = re.compile(r"\b(?:FROM|JOIN)\s+([^\s,;()]+)", re.I)
RE_ALIAS = re.compile(r"\bAS\s+([A-Z_][A-Z0-9_$]*)", re.I)


def _validate_dialect(name: str | None) -> str | None:
    if name is None:
        return DEFAULT_DIALECT

    name = name.lower()
    if name not in [i.lower() for i in DIALECTS]:
        raise ValueError(
            f"Unknown sqlglot dialect '{name}'. "
            f"Choose one of: {', '.join(sorted(DIALECTS))}"
        )
    return name


# ---------------------------------------------------------------------------#
# Helper dataclasses
# ---------------------------------------------------------------------------#


def _tables_in_from(from_clause: exp.From) -> set[str]:
    """
    Return table names that are direct FROM or JOIN sources.
    Ignores tables referenced only in projections or nested sub‑queries.
    """
    names: set[str] = set()

    # Main source after FROM
    if isinstance(from_clause.this, exp.Table):
        names.add(from_clause.this.name)

    # Each JOIN X ...
    for join in from_clause.args.get("joins", []):
        if isinstance(join.this, exp.Table):
            names.add(join.this.name)

    return names


@dataclass
class _TransformState:
    """Keeps all mutable state during a single transformation."""

    counter: int = 0
    mapping: Dict[str, str] = field(default_factory=dict)  # norm_sql -> alias
    cte_definitions: List[str] = field(default_factory=list)
    scalar_aliases: Set[str] = field(default_factory=set)
    used_names: Set[str] = field(default_factory=set)

    def next_alias(self) -> str:
        """Return the next `cte_n` name that doesn't clash with existing names."""
        while True:
            alias = f"cte_{self.counter}"
            self.counter += 1
            if alias.upper() not in self.used_names:
                self.used_names.add(alias.upper())
                return alias


# ---------------------------------------------------------------------------#
# Utility helpers
# ---------------------------------------------------------------------------#

# ---------------------------------------------------------------------------
#  Hoist derived tables  (FROM (SELECT …) alias)  → WITH cte_n AS ( … )
# ---------------------------------------------------------------------------
def _lift_inline_views(root: exp.Expression, state: _TransformState, dialect: str | None):
    """
    • Find every Subquery used as a row‑source (FROM / JOIN).
    • Create or re‑use a CTE name.
    • Replace the Subquery node with Table(cte_name).alias(original_alias).
    """
    for parent in root.find_all((exp.From, exp.Join)):
        sub = parent.this                        # the expression *after* FROM / JOIN
        if not (isinstance(sub, exp.Subquery) and isinstance(sub.this, exp.Select)):
            continue                             # not a derived table

        alias_exp = parent.args.get("alias")
        alias_name = alias_exp.name if alias_exp else None
        if not alias_name:                       # derived table without alias -> skip
            continue

        inner_sql = sub.this.sql(dialect=dialect)
        norm_sql = _normalize_sql(inner_sql)

        if norm_sql not in state.mapping:
            cte_name = state.next_alias()
            state.mapping[norm_sql] = cte_name
            state.cte_definitions.append(_make_cte_sql(inner_sql, cte_name, dialect=dialect))
        else:
            cte_name = state.mapping[norm_sql]

        # Replace Subquery with Table reference, keep original alias
        parent.set("this", exp.Table(this=cte_name))
        # nothing to change for alias; it's already there



def _normalize_sql(sql: str) -> str:
    """Collapse whitespace and upper‑case for dictionary keys."""
    return re.sub(r"\s+", " ", sql).strip().upper()


def _outer_table_aliases(sql: str) -> set[str]:
    """
    Collect table *names* **and** *aliases* that appear in FROM / JOIN clauses.
    Used only to avoid alias collisions and detect correlation.
    """
    names: set[str] = set()

    # ❶ Table names
    for m in RE_FROM_OR_JOIN.finditer(sql):
        names.add(m.group(1).upper())

    # ❷ Aliases — pattern: FROM table  alias   or  JOIN table  alias
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+[^\s,;()]+\s+([A-Z_][A-Z0-9_$]*)", sql, re.I):
        names.add(m.group(1).upper())

    # ❸ Aliases introduced with AS
    names.update(m.group(1).upper() for m in RE_ALIAS.finditer(sql))

    return names


def _is_scalar_query(select_expr: exp.Select) -> bool:
    """
    Return True if the SELECT is judged to be scalar (one row, one column).
    Heuristics are enough for this transformer.
    """
    # Aggregate without GROUP BY
    if select_expr.args.get("group") is None:
        if any(isinstance(node, exp.AggFunc) for node in select_expr.find_all(exp.AggFunc)):
            return True

    # Single constant projection
    items = select_expr.expressions
    if len(items) == 1 and isinstance(items[0], (exp.Literal, exp.Anonymous)):
        return True

    return False


def _make_cte_sql(inner_sql: str, alias: str, *, dialect: str | None) -> str:
    dialect = _validate_dialect(dialect)
    """
    Build `alias AS ( … )`.  If the sub‑query is scalar, force its projection
    to have a column named `val`.
    """
    parsed = sqlglot.parse_one(inner_sql, read=dialect)

    if isinstance(parsed, exp.Select) and _is_scalar_query(parsed):
        proj = parsed.expressions[0]
        if not proj.alias_or_name:
            parsed.set("expressions", [alias_(proj.copy(), "val")])

    return f"{alias} AS ({parsed.sql(dialect=dialect).rstrip(';')})"


# ---------------------------------------------------------------------------#
# 1) sub‑query  ->  CTE
# ---------------------------------------------------------------------------#


def transform_subqueries_to_ctes(
        sql: str,
        *,
        dialect: str | None = None,
) -> str:
    """
    Hoist uncorrelated sub‑queries into WITH‑CTEs.

    * Derived tables  (FROM (SELECT …) alias)     →  WITH cte_n AS ( … )
    * Scalar sub‑queries                          →  WITH cte_n AS ( … ) + .val
    * EXISTS / correlated / set‑ops sub‑queries   →  left untouched
    """

    dialect = _validate_dialect(dialect)

    # ── 1 Pre‑flight ----------------------------------------------------
    statements = sqlparse.parse(sql)
    if len(statements) != 1:
        raise MultipleQueriesError("Exactly one SQL statement is required.")
    raw = sqlparse.format(sql, strip_comments=True).strip()

    state = _TransformState()
    state.used_names.update(_outer_table_aliases(raw))

    # parse once – we’ll mutate the tree
    root = sqlglot.parse_one(raw, read=dialect)

    # ── 2 Hoist *derived tables* (inline views) via AST -----------------
    _lift_inline_views(root, state, dialect)

    # reflect any mutations in the working SQL string
    transformed_sql = root.sql(dialect=dialect)

    # ── 3 Find remaining Parenthesis tokens (scalar / predicate) --------
    parsed_stmt = sqlparse.parse(transformed_sql)[0]
    subqueries: list[sqlparse.sql.Parenthesis] = []

    def _gather(tokens: list[sqlparse.sql.Token]) -> None:
        for tok in tokens:
            if isinstance(tok, sqlparse.sql.Parenthesis):
                if any(
                        t.ttype is sqlparse.tokens.DML and t.value.upper() == "SELECT"
                        for t in tok.tokens
                ):
                    subqueries.append(tok)
            if tok.is_group:
                _gather(list(tok.tokens))

    _gather(list(parsed_stmt.tokens))

    # ── 4 Process scalar / predicate sub‑queries ------------------------
    for sub in subqueries:
        # skip EXISTS (…)
        _, prev_tok = sub.token_prev(-1)
        if prev_tok and prev_tok.ttype is sqlparse_tokens.Keyword and prev_tok.value.upper() == "EXISTS":
            continue

        inner_sql = str(sub)[1:-1].strip()

        if RE_SET_OP.search(inner_sql) or RE_EXISTS.search(inner_sql):
            continue  # set-ops or EXISTS inside the sub‑query

        norm_sql = _normalize_sql(inner_sql)
        alias = state.mapping.get(norm_sql)
        if alias is None:
            select_ast = sqlglot.parse_one(inner_sql, read=dialect)

            # skip scalar+window OR correlated scalar
            if _is_scalar_query(select_ast) and (
                    RE_WINDOW.search(inner_sql)
                    or any(tbl in inner_sql for tbl in state.used_names)
            ):
                continue

            alias = state.next_alias()
            state.mapping[norm_sql] = alias
            state.cte_definitions.append(_make_cte_sql(inner_sql, alias, dialect=dialect))

            if _is_scalar_query(select_ast):
                state.scalar_aliases.add(alias)

        transformed_sql = transformed_sql.replace(str(sub), alias, 1)

    # ── 5 Qualify .val & add CROSS JOINs for scalar CTEs ----------------
    for alias in state.scalar_aliases:
        transformed_sql = re.sub(rf"(?<!\.)\b{alias}\b", f"{alias}.val", transformed_sql)

    if state.scalar_aliases:
        ast_final = sqlglot.parse_one(transformed_sql, read=dialect)
        from_clause: exp.From | None = ast_final.args.get("from")

        if from_clause is None:
            first, *rest = list(state.scalar_aliases)
            ast_final.set("from", exp.From(this=exp.Table(this=first)))
            pending = set(rest)
        else:
            existing = _tables_in_from(from_clause)
            pending = state.scalar_aliases - existing

        for alias in pending:
            ast_final.append("joins", exp.Join(this=exp.Table(this=alias), kind="cross"))

        transformed_sql = ast_final.sql(dialect=dialect)

    # ── 6 Prepend WITH‑clause if any CTEs were created ------------------
    if state.cte_definitions:
        with_clause = "WITH " + ",\n     ".join(state.cte_definitions)
        transformed_sql = f"{with_clause}\n{transformed_sql}"

    # ── 7 Pretty‑print --------------------------------------------------
    return sqlparse.format(transformed_sql, reindent=True, keyword_case="upper")


# ---------------------------------------------------------------------------#
# 2) CTE  ->  sub‑query
# ---------------------------------------------------------------------------#


def transform_ctes_to_subqueries(
        sql: str,
        *,
        dialect: str | None = None,
) -> str:
    """
    Inline every CTE back into the main query.
    Parameters
    ----------
    sql : str
        Input query containing inline sub‑queries.
    dialect : str | None, default = None
        One of sqlglot’s dialect names ("postgres", "bigquery", …).  When
        None we parse with `DEFAULT_DIALECT`, which is currently
        {!r}.format(DEFAULT_DIALECT or "sqlglot’s generic parser")
    """
    dialect = _validate_dialect(dialect)

    ast = sqlglot.parse_one(sql, read=dialect)

    with_clause: exp.With | None = ast.args.get("with")
    if with_clause is None:
        return sql

    # Build alias -> Select mapping
    cte_map: dict[str, exp.Select] = {}
    scalar_aliases: set[str] = set()

    for cte in with_clause.find_all(exp.CTE):
        alias = cte.alias_or_name
        subq: exp.Select = cte.this
        cte_map[alias] = subq
        if _is_scalar_query(subq):
            scalar_aliases.add(alias)

    ast.set("with", None)  # strip WITH‑clause

    # Transformer callback
    def _inline(node: exp.Expression):
        # Drop CROSS JOIN cte_n
        if (
                isinstance(node, exp.Join)
                and node.kind.upper() == "CROSS"
                and isinstance(node.this, exp.Table)
                and node.this.name in cte_map
        ):
            return False

        # FROM/JOIN table
        if isinstance(node, exp.Table) and node.name in cte_map:
            return exp.Subquery(this=cte_map[node.name].copy()).alias(node.alias_or_name)

        # Scalar column cte_n.val ⟶ (SELECT …)
        if isinstance(node, exp.Column) and node.table in scalar_aliases:
            return exp.Subquery(this=cte_map[node.table].copy())

        return node  # keep unchanged

    inlined_ast = ast.transform(_inline)

    return sqlparse.format(
        inlined_ast.sql(dialect=dialect),
        reindent=True,
        keyword_case="upper",
    )
