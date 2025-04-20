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

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Dict, List, Set

import sqlparse
import sqlglot
from sqlglot import exp
from sqlglot.expressions import alias_


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

# ---------------------------------------------------------------------------#
# Helper dataclasses
# ---------------------------------------------------------------------------#


from sqlglot import exp


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


def _normalize_sql(sql: str) -> str:
    """Collapse whitespace and upper‑case for dictionary keys."""
    return re.sub(r"\s+", " ", sql).strip().upper()


def _outer_table_aliases(sql: str) -> Set[str]:
    """
    Collect table names or aliases that appear in FROM / JOIN clauses.
    Used only to avoid alias collisions.
    """
    names = set(m.group(1).upper() for m in RE_FROM_OR_JOIN.finditer(sql))
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


def transform_subqueries_to_ctes(sql: str, *, dialect: str | None = None) -> str:
    """
    Replace uncorrelated sub‑queries with de‑duplicated CTEs.
    Scalar sub‑queries get a `val` column and a `CROSS JOIN cte_n`.
    """
    # -- 1. Strip comments and ensure single statement --------------------
    statements = sqlparse.parse(sql)
    if len(statements) != 1:
        raise MultipleQueriesError("Exactly one SQL statement is required.")
    raw = sqlparse.format(sql, strip_comments=True).strip()

    state = _TransformState()
    state.used_names.update(_outer_table_aliases(raw))

    # -- 2. Find sub‑queries with sqlparse (unchanged) --------------------
    parsed_stmt = statements[0]
    subqueries: list[sqlparse.sql.Parenthesis] = []

    def _gather(tokens: list[sqlparse.sql.Token]) -> None:
        for tok in tokens:
            if isinstance(tok, sqlparse.sql.Parenthesis):
                if any(t.ttype is sqlparse.tokens.DML and t.value.upper() == "SELECT" for t in tok.tokens):
                    subqueries.append(tok)
            if tok.is_group:
                _gather(list(tok.tokens))

    _gather(list(parsed_stmt.tokens))

    transformed_sql = raw

    # -- 3. Replace sub‑queries with aliases, build CTEs  -----------------
    for sub in subqueries:
        inner_sql = str(sub)[1:-1].strip()  # drop outer parens
        if (RE_SET_OP.search(inner_sql)
                or RE_EXISTS.search(inner_sql)
                or RE_WINDOW.search(inner_sql)
                or any(tbl in inner_sql for tbl in state.used_names)):
            continue

        norm_sql = _normalize_sql(inner_sql)
        alias = state.mapping.get(norm_sql)
        if alias is None:
            alias = state.next_alias()
            state.mapping[norm_sql] = alias
            cte_sql = _make_cte_sql(inner_sql, alias, dialect=dialect)
            state.cte_definitions.append(cte_sql)

            select_ast = sqlglot.parse_one(inner_sql, read=dialect)
            if isinstance(select_ast, exp.Select) and _is_scalar_query(select_ast):
                state.scalar_aliases.add(alias)

        transformed_sql = transformed_sql.replace(str(sub), alias, 1)

    # -- 4. Qualify scalar aliases  ---------------------------------------
    for alias in state.scalar_aliases:
        transformed_sql = re.sub(rf"(?<!\.)\b{alias}\b", f"{alias}.val", transformed_sql)

    # -- 5. Insert CROSS JOINs using the AST  -----------------------------
    if state.scalar_aliases:
        select_ast = sqlglot.parse_one(transformed_sql, read=dialect)

        # -----------------------------------------------------------------
        # (a) Guarantee a FROM clause exists
        # -----------------------------------------------------------------
        from_clause: exp.From | None = select_ast.args.get("from")

        if from_clause is None:
            # Pick the first scalar alias as the base row‑source
            first_alias, *rest_aliases = list(state.scalar_aliases)
            from_clause = exp.From(this=exp.Table(this=first_alias))
            select_ast.set("from", from_clause)
            pending_aliases = set(rest_aliases)
            existing_tables = {first_alias}
        else:
            # FROM already present: collect its table and those in top‑level JOINs
            existing_tables = set()
            if isinstance(from_clause.this, exp.Table):
                existing_tables.add(from_clause.this.name)

            for j in select_ast.args.get("joins", []):
                if isinstance(j.this, exp.Table):
                    existing_tables.add(j.this.name)

            pending_aliases = state.scalar_aliases - existing_tables

        # -----------------------------------------------------------------
        # (b) Append CROSS JOINs for any still‑missing scalar CTEs
        # -----------------------------------------------------------------
        for alias in pending_aliases:
            select_ast.append(
                "joins",
                exp.Join(this=exp.Table(this=alias), kind="cross"),
            )

        # -----------------------------------------------------------------
        transformed_sql = select_ast.sql(dialect=dialect)

    # -- 6. Prepend WITH‑clause if any CTEs were created ------------------
    if state.cte_definitions:
        with_clause = "WITH " + ",\n     ".join(state.cte_definitions)
        transformed_sql = f"{with_clause}\n{transformed_sql}"

    # -- 7. Pretty‑print and return --------------------------------------
    return sqlparse.format(transformed_sql, reindent=True, keyword_case="upper")


# ---------------------------------------------------------------------------#
# 2) CTE  ->  sub‑query
# ---------------------------------------------------------------------------#


def transform_ctes_to_subqueries(sql: str, *, dialect: str | None = None) -> str:
    """
    Inline every CTE back into the main query.
    """
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
