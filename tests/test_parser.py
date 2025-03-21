import pytest
from query_parser import (
    transform_query_to_ctes,
    transform_ctes_to_subqueries,
    remove_comments,
    MultipleQueriesError,
    get_outer_tables
)


def normalize(sql: str) -> str:
    """Normalize SQL by uppercasing and collapsing whitespace."""
    return " ".join(sql.upper().split())


# --- Tests for remove_comments() ---
@pytest.mark.parametrize("query, expected_fragment", [
    (
            "-- This is a single-line comment\nSELECT a, b FROM table1 -- end comment",
            "SELECT A, B FROM TABLE1"
    ),
    (
            "/* multi-line\n comment */ SELECT a FROM table1 /* another comment */",
            "SELECT A FROM TABLE1"
    ),
])
def test_remove_comments(query: str, expected_fragment: str) -> None:
    result: str = remove_comments(query)
    assert normalize(result) == normalize(expected_fragment)


# --- Tests for get_outer_tables() ---
@pytest.mark.parametrize("query, expected_tables", [
    (
            "SELECT t1.a, (SELECT t2.b FROM table2 t2) FROM table1 t1",
            {"TABLE1"}
    ),
    (
            "SELECT a FROM table1 JOIN table2 ON table1.id = table2.id",
            {"TABLE1", "TABLE2"}
    ),
    (
            """
            SELECT t1.a,
                   (SELECT t2.b FROM table2 t2 WHERE t2.id IN (SELECT id FROM table3))
            FROM table1 t1
            JOIN table4 t4 ON t1.id = t4.id
            WHERE t1.status = 'active'
            """,
            {"TABLE1", "TABLE4"}
    ),
])
def test_get_outer_tables(query: str, expected_tables: set) -> None:
    outer_tables = get_outer_tables(query)
    normalized_tables = {table.upper() for table in outer_tables}
    assert expected_tables.issubset(normalized_tables)


# --- Tests for transform_query_to_ctes() ---
@pytest.mark.parametrize("query, expected_substring", [
    # Query with no subqueries should remain essentially unchanged.
    (
            "SELECT a, b FROM table1",
            "SELECT A, B FROM TABLE1"
    ),
    # Query with an uncorrelated subquery should produce a WITH clause.
    (
            """
            SELECT a, b,
                   (SELECT MAX(c) FROM table2) as max_c
            FROM table1
            """,
            "WITH"
    ),
    # Correlated subquery: the inner subquery remains unchanged.
    (
            """
            SELECT a, b,
                   (SELECT MAX(c) FROM table2 WHERE table2.id = table1.id) as max_c
            FROM table1
            """,
            "(SELECT MAX(C) FROM TABLE2 WHERE TABLE2.ID = TABLE1.ID)"
    ),
    # Scalar subquery in WHERE clause should be transformed with a CROSS JOIN.
    (
            """
            SELECT a
            FROM table1
            WHERE a > (SELECT AVG(b) FROM table2)
            """,
            "CROSS JOIN CTE_0"
    ),
    # Uncorrelated subquery in JOIN clause should be transformed.
    (
            """
            SELECT t1.a, t2.max_val
            FROM table1 t1
            JOIN (SELECT id, MAX(val) as max_val FROM table2 GROUP BY id) t2 ON t1.id = t2.id
            """,
            "WITH"
    ),
])
def test_transform_subquery(query: str, expected_substring: str) -> None:
    result: str = transform_query_to_ctes(query)
    assert expected_substring.upper() in normalize(result)


@pytest.mark.parametrize("query, expected_alias", [
    (
            """
            SELECT a,
                   (SELECT MAX(c) FROM table2) as max_c,
                   (SELECT MAX(c) FROM table2) as another_max
            FROM table1
            """,
            "CTE_0"
    ),
])
def test_duplicate_subqueries(query: str, expected_alias: str) -> None:
    result: str = transform_query_to_ctes(query)
    # Check that the CTE is defined only once.
    assert normalize(result).count(normalize(f"{expected_alias} AS (")) == 1
    # And that the alias appears (qualified) in the outer query.
    assert normalize(result).count(expected_alias.upper()) >= 3


@pytest.mark.parametrize("query, expected_fragment", [
    (
            """
            SELECT a, b,
                   (SELECT c FROM table2 UNION SELECT c FROM table3) as union_c
            FROM table1
            """,
            "UNION"
    ),
    (
            """
            SELECT a
            FROM table1
            WHERE EXISTS (SELECT 1 FROM table2 WHERE table2.id = table1.id)
            """,
            "EXISTS"
    ),
    (
            """
            SELECT a,
                   (SELECT AVG(val) OVER (PARTITION BY id) FROM table2) as avg_val
            FROM table1
            """,
            "OVER ("
    ),
])
def test_non_transformable_subqueries(query: str, expected_fragment: str) -> None:
    result: str = transform_query_to_ctes(query)
    # Expect no transformation: the query should not start with WITH.
    assert not normalize(result).startswith("WITH")
    assert expected_fragment.upper() in normalize(result)


@pytest.mark.parametrize("query", [
    "SELECT a FROM table1; SELECT b FROM table2;",
])
def test_multiple_queries_exception(query: str) -> None:
    with pytest.raises(MultipleQueriesError):
        transform_query_to_ctes(query)


def test_get_outer_tables_nested() -> None:
    query: str = """
    SELECT t1.a,
           (SELECT t2.b FROM table2 t2 WHERE t2.id IN (SELECT id FROM table3)) as b_val
    FROM table1 t1
    JOIN table4 t4 ON t1.id = t4.id
    WHERE t1.status = 'active'
    """
    outer_tables: set = get_outer_tables(query)
    normalized_tables = {table.upper() for table in outer_tables}
    assert "TABLE1" in normalized_tables
    assert "TABLE4" in normalized_tables
    assert "TABLE2" not in normalized_tables
    assert "TABLE3" not in normalized_tables


def test_query_formatting() -> None:
    query: str = "SELECT a,b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)"
    result: str = transform_query_to_ctes(query)
    # Check that line breaks exist and that the normalized output starts with WITH or SELECT.
    assert "\n" in result
    assert normalize(result).startswith("WITH") or normalize(result).startswith("SELECT")


def test_full_output_scalar_where() -> None:
    """
    Test a query with a scalar subquery in the WHERE clause.
    Expected transformation:

    WITH cte_0 AS (
        SELECT AVG(b) AS val FROM table2
    )
    SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val
    """
    query = "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)"
    expected = (
        "WITH cte_0 AS ("
        "SELECT AVG(b) AS val FROM table2"
        ") "
        "SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val"
    )
    output = transform_query_to_ctes(query)
    assert normalize(output) == normalize(expected)


def test_full_output_duplicate_scalar() -> None:
    """
    Test a query where the same scalar subquery appears twice.
    Expected transformation:

    WITH cte_0 AS (
        SELECT MAX(c) AS val FROM table2
    )
    SELECT a, cte_0.val AS max_c, cte_0.val AS another_max FROM table1 CROSS JOIN cte_0
    """
    query = (
        "SELECT a, "
        "(SELECT MAX(c) FROM table2) as max_c, "
        "(SELECT MAX(c) FROM table2) as another_max "
        "FROM table1"
    )
    expected = (
        "WITH cte_0 AS ("
        "SELECT MAX(c) AS val FROM table2"
        ") "
        "SELECT a, cte_0.val AS max_c, cte_0.val AS another_max "
        "FROM table1 CROSS JOIN cte_0"
    )
    output = transform_query_to_ctes(query)
    assert normalize(output) == normalize(expected)


def test_full_output_correlated_subquery() -> None:
    """
    Test a query with a correlated subquery. In this case, no transformation should occur.
    """
    query = (
        "SELECT a, b FROM table1 WHERE a > "
        "(SELECT AVG(b) FROM table2 WHERE table2.id = table1.id)"
    )
    # Expected output is identical to input (after formatting).
    expected = query
    output = transform_query_to_ctes(query)
    assert normalize(output) == normalize(expected)


def test_full_output_window_function() -> None:
    """
    Test a query with a subquery that contains a window function.
    Because of the window function, the query should remain unchanged.
    """
    query = (
        "SELECT a, (SELECT AVG(val) OVER (PARTITION BY id) FROM table2) as avg_val "
        "FROM table1"
    )
    expected = query
    output = transform_query_to_ctes(query)
    assert normalize(output) == normalize(expected)


# --- New tests for transform_ctes_to_subqueries() --- #

def test_transform_ctes_to_subqueries_no_with() -> None:
    """
    When the query does not start with a WITH clause,
    transform_ctes_to_subqueries() should return the original query.
    """
    query = "SELECT a, b FROM table1 WHERE a > 10"
    result = transform_ctes_to_subqueries(query)
    assert normalize(result) == normalize(query)


@pytest.mark.parametrize("query, expected", [
    (
            # Basic scalar subquery transformation:
            "WITH cte_0 AS (SELECT AVG(b) AS val FROM table2) "
            "SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val",
            "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS VAL FROM table2)"
    ),
])
def test_transform_ctes_to_subqueries_basic(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)
    assert normalize(result) == normalize(expected)


@pytest.mark.parametrize("query, expected", [
    (
            # Duplicate scalar subqueries transformed back inline:
            "WITH cte_0 AS (SELECT MAX(c) AS val FROM table2) "
            "SELECT a, cte_0.val AS max_c, cte_0.val AS another_max FROM table1 CROSS JOIN cte_0",
            "SELECT a, (SELECT MAX(c) AS VAL FROM table2) AS max_c, (SELECT MAX(c) AS VAL  FROM table2) AS another_max FROM table1"
    ),
])
def test_transform_ctes_to_subqueries_duplicate(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)
    assert normalize(result) == normalize(expected)


@pytest.mark.parametrize("query, expected", [
    (
            # Multiple CTEs (one scalar and one non-scalar) transformed back to inline subqueries.
            "WITH cte_0 AS (SELECT AVG(b) AS val FROM table2), "
            "cte_1 AS (SELECT MAX(c) AS val FROM table3) "
            "SELECT a, b FROM table1 CROSS JOIN cte_0 CROSS JOIN cte_1 "
            "WHERE a > cte_0.val AND b = cte_1",
            "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS VAL FROM table2) AND b = (SELECT MAX(c) AS VAL FROM table3)"
    ),
])
def test_transform_ctes_to_subqueries_multiple(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)
    assert normalize(result) == normalize(expected)
