# tests/test_sql_subquery_cte_transformer.py

import pytest
from collections import Counter

import duckdb
import sqlglot
from sqlglot import parse_one, exp

from query_parser import (
    transform_subqueries_to_ctes,
    transform_ctes_to_subqueries,
    MultipleQueriesError,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def assert_sql_equal(actual: str, expected: str) -> None:
    """
    Assert that two SQL strings produce the same AST.
    """
    a = parse_one(actual)
    b = parse_one(expected)
    assert a == b, f"\nACTUAL:   {a}\nEXPECTED: {b}"


def _rows(con: duckdb.DuckDBPyConnection, sql: str):
    """Execute *sql* on *con* and return a multiset of rows (order-independent)."""
    return Counter(tuple(row) for row in con.execute(sql).fetchall())


# ---------------------------------------------------------------------------
# DuckDB fixture – in-memory database used for semantic checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE table1 (a INT, b INT, id INT, status VARCHAR);
        INSERT INTO table1 VALUES
          (1, 10, 1, 'active'),
          (2, 20, 2, 'inactive'),
          (3, 30, 1, 'active'),
          (4, 40, 3, 'inactive');
        """
    )
    con.execute(
        """
        CREATE TABLE table2 (b INT, c INT, id INT, val INT);
        INSERT INTO table2 VALUES
          (10, 100, 1, 5),
          (20, 200, 2, 10),
          (30, 300, 1, 15),
          (40, 400, 3, 20);
        """
    )
    con.execute("CREATE TABLE table3(id INT);")
    con.execute("INSERT INTO table3 VALUES (1), (2), (3);")
    con.execute("CREATE TABLE table4(id INT);")
    con.execute("INSERT INTO table4 VALUES (1), (2), (4);")
    yield con
    con.close()


# ---------------------------------------------------------------------------
# transform_subqueries_to_ctes – AST/text behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT a, b FROM table1", "SELECT a, b FROM table1"),
        (
            """
            SELECT a,
                   b,
                   (SELECT MAX(c) FROM table2) AS max_c
            FROM table1
            """,
            """
            WITH cte_0 AS (SELECT MAX(c) AS val FROM table2)
            SELECT a, b, cte_0.val AS max_c
            FROM table1 CROSS JOIN cte_0
            """,
        ),
        (
            """
            SELECT a, b
            FROM table1
            WHERE a > (SELECT AVG(b) FROM table2)
            """,
            """
            WITH cte_0 AS (SELECT AVG(b) AS val FROM table2)
            SELECT a, b
            FROM table1 CROSS JOIN cte_0
            WHERE a > cte_0.val
            """,
        ),
        (
            """
            SELECT a,
                   (SELECT MAX(c) FROM table2) AS max_c,
                   (SELECT MAX(c) FROM table2) AS another_max
            FROM table1
            """,
            """
            WITH cte_0 AS (SELECT MAX(c) AS val FROM table2)
            SELECT a, cte_0.val AS max_c, cte_0.val AS another_max
            FROM table1 CROSS JOIN cte_0
            """,
        ),
    ],
)
def test_transform_subqueries_to_ctes_text(query: str, expected: str) -> None:
    assert_sql_equal(transform_subqueries_to_ctes(query), expected)


# ---------------------------------------------------------------------------
# Semantic equivalence using DuckDB – subquery ➜ CTE
# ---------------------------------------------------------------------------

_semantic_cases_sub_to_cte = [
    "SELECT a, b FROM table1",
    "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)",
    """
    SELECT a,
           (SELECT MAX(c) FROM table2) AS max_c,
           (SELECT MAX(c) FROM table2) AS another_max
    FROM table1
    """,
]

@pytest.mark.parametrize("query", _semantic_cases_sub_to_cte)
def test_semantic_equivalence_sub_to_cte(query: str, db):
    transformed = transform_subqueries_to_ctes(query)
    assert _rows(db, query) == _rows(db, transformed)


# ---------------------------------------------------------------------------
# Idempotency – running transform twice yields same AST
# ---------------------------------------------------------------------------

def test_transform_is_idempotent():
    query = """
        WITH cte_0 AS (SELECT AVG(b) AS val FROM table2)
        SELECT a
        FROM table1 CROSS JOIN cte_0
        WHERE a > cte_0.val
    """
    once = transform_ctes_to_subqueries(query)
    twice = transform_ctes_to_subqueries(once)
    assert_sql_equal(once, twice)


# ---------------------------------------------------------------------------
# transform_ctes_to_subqueries – AST/text behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT a, b FROM table1 WHERE a > 10", "SELECT a, b FROM table1 WHERE a > 10"),
        (
            """
            WITH cte_0 AS (SELECT AVG(b) AS val FROM table2)
            SELECT a, b
            FROM table1 CROSS JOIN cte_0
            WHERE a > cte_0.val
            """,
            "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS val FROM table2)",
        ),
        (
            """
            WITH cte_0 AS (SELECT MAX(c) AS val FROM table2)
            SELECT a, cte_0.val AS max_c, cte_0.val AS another_max
            FROM table1 CROSS JOIN cte_0
            """,
            "SELECT a, (SELECT MAX(c) AS val FROM table2) AS max_c, (SELECT MAX(c) AS val FROM table2) AS another_max FROM table1",
        ),
    ],
)
def test_transform_ctes_to_subqueries_text(query: str, expected: str):
    assert_sql_equal(transform_ctes_to_subqueries(query), expected)


# ---------------------------------------------------------------------------
# Deterministic CTE names
# ---------------------------------------------------------------------------

def test_sequential_cte_names():
    query = "SELECT (SELECT 1), (SELECT 2)"
    ast = parse_one(transform_subqueries_to_ctes(query))
    # Expect two distinct CTEs in the WITH clause
    with_clause = ast.args.get("with")
    assert isinstance(with_clause, exp.With)
    names = [cte.alias_or_name for cte in with_clause.find_all(exp.CTE)]
    assert names == ["cte_0", "cte_1"]


# ---------------------------------------------------------------------------
# Error handling – multiple statements should raise
# ---------------------------------------------------------------------------

def test_multiple_queries_raises():
    with pytest.raises(MultipleQueriesError):
        transform_subqueries_to_ctes("SELECT 1; SELECT 2;")
